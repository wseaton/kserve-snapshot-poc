//! A fixed number of threads, each with a resource of its own that is made
//! when the lane first has work and kept.
use std::sync::{Arc, Mutex, mpsc};

use anyhow::{Context, Result, anyhow};

type Job<R> = Box<dyn FnOnce(Result<&R>) -> bool + Send>;

pub struct Lanes<R> {
    jobs: mpsc::Sender<Job<R>>,
}

impl<R> Clone for Lanes<R> {
    fn clone(&self) -> Self {
        Self {
            jobs: self.jobs.clone(),
        }
    }
}

impl<R: 'static> Lanes<R> {
    /// `make` is called on a lane's own thread, with the lane's number.
    pub fn start(
        count: usize,
        make: impl Fn(usize) -> Result<R> + Send + Sync + 'static,
    ) -> Result<Self> {
        let (jobs, queue) = mpsc::channel::<Job<R>>();
        let (queue, make) = (Arc::new(Mutex::new(queue)), Arc::new(make));
        for lane in 0..count {
            let (queue, make) = (Arc::clone(&queue), Arc::clone(&make));
            std::thread::Builder::new()
                .name(format!("lane-{lane}"))
                .spawn(move || {
                    let mut resource = None;
                    loop {
                        let next = match queue.lock() {
                            Ok(queue) => queue.recv(),
                            Err(_) => return,
                        };
                        let Ok(job) = next else { return };
                        if resource.is_none() {
                            match make(lane) {
                                Ok(made) => resource = Some(made),
                                Err(error) => {
                                    job(Err(error));
                                    continue;
                                }
                            }
                        }
                        if let Some(ready) = &resource
                            && !job(Ok(ready))
                        {
                            resource = None;
                        }
                    }
                })
                .context("start a lane")?;
        }
        Ok(Self { jobs })
    }

    /// Run `work` on a free lane and wait for it. The resource of a lane
    /// whose work failed is dropped and made again for its next work.
    pub fn run<T: Send + 'static>(
        &self,
        work: impl FnOnce(&R) -> Result<T> + Send + 'static,
    ) -> Result<T> {
        let (done, outcome) = mpsc::channel();
        let job: Job<R> = Box::new(move |resource| {
            let result = resource.and_then(work);
            let fine = result.is_ok();
            let _ = done.send(result);
            fine
        });
        self.jobs
            .send(job)
            .map_err(|_| anyhow!("every lane has stopped"))?;
        outcome.recv().context("a lane dropped its work")?
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier};
    use std::time::Duration;

    use anyhow::anyhow;

    use crate::lanes::Lanes;

    #[test]
    fn as_many_jobs_run_at_once_as_there_are_lanes() {
        let lanes = Lanes::start(3, Ok).unwrap();
        let together = Arc::new(Barrier::new(3));
        let callers: Vec<_> = (0..3)
            .map(|_| {
                let (lanes, together) = (lanes.clone(), Arc::clone(&together));
                std::thread::spawn(move || {
                    lanes.run(move |lane: &usize| {
                        together.wait();
                        Ok(*lane)
                    })
                })
            })
            .collect();
        let mut used: Vec<usize> = callers
            .into_iter()
            .map(|caller| caller.join().unwrap().unwrap())
            .collect();
        used.sort();
        assert_eq!(used, [0, 1, 2]);
    }

    #[test]
    fn a_lane_makes_its_resource_once_and_again_only_after_a_failure() {
        let made = Arc::new(AtomicUsize::new(0));
        let count = Arc::clone(&made);
        let lanes = Lanes::start(1, move |_| Ok(count.fetch_add(1, Ordering::SeqCst))).unwrap();
        assert_eq!(made.load(Ordering::SeqCst), 0);
        assert_eq!(lanes.run(|resource: &usize| Ok(*resource)).unwrap(), 0);
        assert_eq!(lanes.run(|resource: &usize| Ok(*resource)).unwrap(), 0);
        let failed =
            lanes.run(|_: &usize| -> anyhow::Result<()> { Err(anyhow!("the peer went away")) });
        assert!(format!("{:#}", failed.unwrap_err()).contains("the peer went away"));
        assert_eq!(lanes.run(|resource: &usize| Ok(*resource)).unwrap(), 1);
        assert_eq!(made.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn a_resource_that_cannot_be_made_fails_the_work_and_is_tried_again() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let count = Arc::clone(&attempts);
        let lanes = Lanes::start(1, move |_| match count.fetch_add(1, Ordering::SeqCst) {
            0 => Err(anyhow!("no memory")),
            attempt => Ok(attempt),
        })
        .unwrap();
        let first = lanes.run(|resource: &usize| Ok(*resource));
        assert!(format!("{:#}", first.unwrap_err()).contains("no memory"));
        assert_eq!(lanes.run(|resource: &usize| Ok(*resource)).unwrap(), 1);
    }

    #[test]
    fn work_waits_for_a_lane_when_all_are_taken() {
        let lanes = Lanes::start(1, Ok).unwrap();
        let order = Arc::new(std::sync::Mutex::new(Vec::new()));
        let (slow, first) = (lanes.clone(), Arc::clone(&order));
        let holder = std::thread::spawn(move || {
            slow.run(move |_: &usize| {
                std::thread::sleep(Duration::from_millis(200));
                first.lock().unwrap().push("first");
                Ok(())
            })
        });
        std::thread::sleep(Duration::from_millis(50));
        let second = Arc::clone(&order);
        lanes
            .run(move |_: &usize| {
                second.lock().unwrap().push("second");
                Ok(())
            })
            .unwrap();
        holder.join().unwrap().unwrap();
        assert_eq!(*order.lock().unwrap(), ["first", "second"]);
    }
}

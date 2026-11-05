"""exec a command with io_uring_setup returning ENOSYS, like Dynamo's block-iouring seccomp profile."""
import ctypes as C, os, struct, sys

AUDIT_ARCH_X86_64 = 0xC000003E
NR_IO_URING_SETUP = 425
BPF_LD_W_ABS, BPF_JEQ_K, BPF_RET_K = 0x20, 0x15, 0x06
SECCOMP_RET_ALLOW, SECCOMP_RET_ERRNO, SECCOMP_RET_KILL = 0x7FFF0000, 0x00050000, 0
ENOSYS = 38
prog = [
    (BPF_LD_W_ABS, 0, 0, 4),                       # arch
    (BPF_JEQ_K, 1, 0, AUDIT_ARCH_X86_64),
    (BPF_RET_K, 0, 0, SECCOMP_RET_KILL),
    (BPF_LD_W_ABS, 0, 0, 0),                       # syscall nr
    (BPF_JEQ_K, 0, 1, NR_IO_URING_SETUP),
    (BPF_RET_K, 0, 0, SECCOMP_RET_ERRNO | ENOSYS),
    (BPF_RET_K, 0, 0, SECCOMP_RET_ALLOW),
]
buf = C.create_string_buffer(b"".join(struct.pack("HBBI", *i) for i in prog))
class SockFprog(C.Structure):
    _fields_ = [("len", C.c_ushort), ("filter", C.c_void_p)]
fprog = SockFprog(len(prog), C.cast(buf, C.c_void_p))
libc = C.CDLL(None, use_errno=True)
PR_SET_NO_NEW_PRIVS, PR_SET_SECCOMP, SECCOMP_MODE_FILTER = 38, 22, 2
assert libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) == 0, C.get_errno()
assert libc.prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, C.byref(fprog), 0, 0) == 0, C.get_errno()
os.execvp(sys.argv[1], sys.argv[1:])

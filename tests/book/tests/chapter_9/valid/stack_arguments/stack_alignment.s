    .globl _main
_main:
    pushq %rbp
    movq  %rsp, %rbp
    subq $160, %rsp
    movl $3, -8(%rbp)
    movl $1, -16(%rbp)
    movl $2, -24(%rbp)
    movl $3, -32(%rbp)
    movl $4, -40(%rbp)
    movl $5, -48(%rbp)
    movl $6, -56(%rbp)
    movl $7, -64(%rbp)
    movl $8, -72(%rbp)
    movl -16(%rbp), %edi
    movl -24(%rbp), %esi
    movl -32(%rbp), %edx
    movl -40(%rbp), %ecx
    movl -48(%rbp), %r8d
    movl -56(%rbp), %r9d
    pushq -72(%rbp)
    pushq -64(%rbp)
    call _even_arguments
    addq $16, %rsp
    movl %eax, -80(%rbp)
    movl $1, -88(%rbp)
    movl $2, -96(%rbp)
    movl $3, -104(%rbp)
    movl $4, -112(%rbp)
    movl $5, -120(%rbp)
    movl $6, -128(%rbp)
    movl $7, -136(%rbp)
    movl $8, -144(%rbp)
    movl $9, -152(%rbp)
    subq $8, %rsp
    movl -88(%rbp), %edi
    movl -96(%rbp), %esi
    movl -104(%rbp), %edx
    movl -112(%rbp), %ecx
    movl -120(%rbp), %r8d
    movl -128(%rbp), %r9d
    pushq -152(%rbp)
    pushq -144(%rbp)
    pushq -136(%rbp)
    call _odd_arguments
    addq $32, %rsp
    movl %eax, -160(%rbp)
    movl -8(%rbp), %eax
    movq %rbp, %rsp
    popq %rbp
    ret

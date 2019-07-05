program escreve_sin_cos

    implicit none
    
    !Declarar variáveis
    integer :: i, nx
    real(4) :: xi, xf, dx
    real(4), allocatable, dimension(:) :: x, sinx, cosx

    nx=200
    xi = 0.
    xf = 3.14

    !Alocar variáveis
    allocate(x(nx), sinx(nx), cosx(nx))

    !Calcular
    dx = (xi-xf)/(nx-1)
    do i=1, nx
        x(i) = real(i-1,4)*dx
        sinx(i) = sin(x(i))
        cosx(i) = cos(x(i))
    end do

    !Escrever em disco e abrir em outro
    !programa para gerar imagens
    open (1, file = "teste.csv")
    do i=1, nx
        write(1,*) x(i), sinx(i), cosx(i)
    end do
    close(1)
    
    !Encerrar programa
end program escreve_sin_cos

program fbound

    integer, dimension(10), target :: X = (/0,1,2,3,4,5,6,7,8,9/)
    integer, dimension(:), pointer :: P1

    X = 1

    P1 => X
    print *, P1(1)

    call rebound(X)

contains

    subroutine rebound(X)
        
        integer, dimension(-2:) :: X

        print *, X(-2)

    end subroutine
end program

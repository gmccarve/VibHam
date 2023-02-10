C FILE: HamilF.f90

        subroutine Harmonic(V, NU)

C
C       Subroutine used to populate the Harmonic portion of a Hamiltonian matrix
C
C       Variables:
C                V    - Hamiltonian matrix dimension
C                NU   - Harmonic frequency (s^-1)
C                HBAR - Reduced Planck constant
C
C            Returns:
C                H - The harmonic Hamiltonain Matrix


        INTEGER :: V, I
        DOUBLE PRECISION :: NU, HBAR, H(0:V,0:V)

        PARAMETER (HBAR = 1.054571817E-34)

        H = 0.d0

        DO I=0,V
          H(I,I) = (2.d0 * I + 1.d0) * HBAR * NU * 1./2.
        ENDDO

        OPEN (unit=12, file='harmonic.tmp', status='REPLACE')
        WRITE (12,*) H
        CLOSE (unit=12)

        END

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

        subroutine Anharmonic(V, COEF, BETA)

C       Function used to populate the Anharmonic Hamiltonian matrix.
C            These integrals are calculated analytically. 
C
C            Variables:
C                V       - Hamiltonian matrix dimension
C                COEF    - Array of power series expansion coefficients
C                BETA    - Beta value for the diatomic molecule
C                H       - Hamiltonian Matrix
C                Hh      - temporary Hamiltonian Matrix
C                XP      - Order of power series expansion coefficient
C                C       - Power series expansion coefficient
C                NORMI   - Normalization consant for bra
C                NORMJ   - Normalization consant for ket
C                HERMI   - Hermite polynomial array for bra
C                HERMJ   - Hermite polynomial array for ket
C                INTE    - running value of integral
C                TXP     - Total exponent given __Hermite polynomials of bra, ket, and power series expansion
C                HVAL    - Total value of bra and ket __Hermite polynomial coefficients
C                BVAL    - Value to account for powers of beta^n
C                FVAL    - Value of factorials given analtical solution to integral
C
C            Returns:
C                H       - The anharmonic Hamiltonian matrix
C

        interface
          function Hermite(X) RESULT(H)
            INTEGER :: X
            DOUBLE PRECISION :: H(X)
          end function
          function Norm(BETA,V) RESULT (N)
            DOUBLE PRECISION :: BETA, N
            INTEGER :: V
          end function
          function Fac(N) RESULT(P)
            INTEGER :: N
            DOUBLE PRECISION :: P
          end function
          function Ex(N) RESULT(X)
            INTEGER :: N
            DOUBLE PRECISION :: X
          end function
        end interface


        INTEGER :: V, N, NC(1), NV, NVV

        DOUBLE PRECISION :: COEF(:), C, BETA

        DOUBLE PRECISION :: HERMI(0:V), HERMJ(0:V)
        INTEGER :: XP, TXP, I, J
        DOUBLE PRECISION :: NORMI, NORMJ
        DOUBLE PRECISION :: INTE
        DOUBLE PRECISION :: HVAL, BVAL, FVAL

        DOUBLE PRECISION :: H(0:V,0:V), Hh(0:V,0:V)

        DOUBLE PRECISION :: HBAR, ANG2M, PI
        PARAMETER (HBAR = 1.054571817E-34)
        PARAMETER (ANG2M = 1E-10)
        PARAMETER (PI = 4 * ATAN(1.d0))

        H = 0.d0

        NC = SHAPE(COEF)

        NORMI = 0.d0
        NORMJ = 0.d0

        DO N=2,NC(1)
          Hh = 0.d0
          XP = N+1
          C = COEF(N) * ANG2M**XP

          BVAL = (1. / BETA**(XP+1))

          HERMI = 0

          DO NV=0,V
            HERMI = Hermite(NV)
            NORMI = Norm(BETA, NV)

            HERMJ = 0

            DO NVV=NV,V
              HERMJ = Hermite(NVV)
              NORMJ = Norm(BETA, NVV)

              INTE = 0.d0

              DO I=0,NV
                IF (HERMI(I) .NE. 0) THEN
                  DO J=0,NVV
                    IF (HERMJ(J) .NE. 0) THEN
                      TXP = I + J + XP

                      IF (MOD(TXP,2) .EQ. 0) THEN
                        HVAL = HERMI(I) * HERMJ(J)
                        FVAL = Fac(TXP) / (Ex(TXP) * Fac(TXP/2))
                        INTE = INTE + HVAL * FVAL * BVAL * SQRT(PI)

                      ENDIF
                    ENDIF
                  ENDDO
                ENDIF
              ENDDO

              Hh(NV, NVV) = INTE * NORMI * NORMJ * C
              Hh(NVV, NV) = INTE * NORMI * NORMJ * C

            ENDDO
          ENDDO

          H = H + Hh

        ENDDO
        
        OPEN (unit=12, file='anharmonic.tmp', status='REPLACE')
        WRITE (12,*) H
        CLOSE (unit=12)

        END

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

        subroutine TDM(V, COEF, BETA)

C       Function used to populate the Transition Dipole Moment Hamiltonian matrix.
C            These integrals are calculated analytically. 
C
C            Variables:
C                V       - Hamiltonian matrix dimension
C                COEF    - Array of power series expansion coefficients
C                BETA    - Beta value for the diatomic molecule
C                H       - Hamiltonian Matrix
C                Hh      - temporary Hamiltonian Matrix
C                XP      - Order of power series expansion coefficient
C                MU       - Power series expansion coefficient
C                NORMI   - Normalization consant for bra
C                NORMJ   - Normalization consant for ket
C                HERMI   - Hermite polynomial array for bra
C                HERMJ   - Hermite polynomial array for ket
C                INTE    - running value of integral
C                TXP     - Total exponent given __Hermite polynomials of bra, ket, and power series expansion
C                HVAL    - Total value of bra and ket __Hermite polynomial coefficients
C                BVAL    - Value to account for powers of beta^n
C                FVAL    - Value of factorials given analtical solution to integral
C
C            Returns:
C                H       - The anharmonic Hamiltonian matrix
C


        interface
          function Hermite(X) RESULT(H)
            INTEGER :: X
            DOUBLE PRECISION :: H(X)
          end function
          function Norm(BETA,V) RESULT (N)
            DOUBLE PRECISION :: BETA, N
            INTEGER :: V
          end function
          function Fac(N) RESULT(P)
            INTEGER :: N
            DOUBLE PRECISION :: P
          end function
          function Ex(N) RESULT(X)
            INTEGER :: N
            DOUBLE PRECISION :: X
          end function
        end interface


        INTEGER :: V, N, NC(1), NV, NVV

        DOUBLE PRECISION :: COEF(:), MU, BETA

        DOUBLE PRECISION :: HERMI(0:V), HERMJ(0:V)
        INTEGER :: TXP, I, J
        DOUBLE PRECISION :: NORMI, NORMJ
        DOUBLE PRECISION :: INTE
        DOUBLE PRECISION :: HVAL, BVAL, FVAL

        DOUBLE PRECISION :: H(0:V,0:V)

        DOUBLE PRECISION :: HBAR, ANG2M, PI
        PARAMETER (HBAR = 1.054571817E-34)
        PARAMETER (ANG2M = 1E-10)
        PARAMETER (PI = 4 * ATAN(1.d0))

        H = 0.d0

        NC = SHAPE(COEF)

        NORMI = 0.d0
        NORMJ = 0.d0

        DO NV=0,V
          HERMI = Hermite(NV)
          NORMI = Norm(BETA, NV)

          HERMJ = 0

          DO NVV=NV,V
            HERMJ = Hermite(NVV)
            NORMJ = Norm(BETA, NVV)

            INTE = 0.d0

            DO N=NC(1),1,-1
              MU = COEF(N) * ANG2M**(NC(1)-N)

              BVAL = (1. / BETA**(NC(1)-N+1))

              DO I=0,NV
                IF (HERMI(I) .NE. 0) THEN
                  DO J=0,NVV
                    IF (HERMJ(J) .NE. 0) THEN
                      TXP = I + J + (NC(1)-N)

                      IF (MOD(TXP,2) .EQ. 0) THEN
                        HVAL = HERMI(I) * HERMJ(J)
                        FVAL = Fac(TXP) / (Ex(TXP) * Fac(TXP/2))
                        INTE = INTE + MU * HVAL * FVAL * BVAL * SQRT(PI)

                      ENDIF
                    ENDIF
                  ENDDO
                ENDIF
              ENDDO

            ENDDO

            H(NV, NVV) = INTE * NORMI * NORMJ
            H(NVV, NV) = INTE * NORMI * NORMJ

          ENDDO
        ENDDO
        
        OPEN (unit=12, file='tdm.tmp', status='REPLACE')
        WRITE (12,*) H
        CLOSE (unit=12)

        END

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

        subroutine Centrifugal(TRAP,V,BETA,REQ)

C       Function used to populate the centrifugal potential Hamiltonian Matrix.
C            These integrals are solved numerically using the trapezoid rule
C
C            Variables:
C                Trap                - Number of intervals used for trapezoid rule
C                self.maxJ           - Maximum rotational quantum number
C                slef.maxV           - Maximum vibrational quantum number
C                self.rEq            - Equilibrium bond distance
C                self.beta           - Beta value for the diatomic molecule
C                self.reduced_mass   - Reduced mass for the diatomic molecule
C                old_mat             - Old centrifugal matrix to be read in
C                J_Factor            - J(J+1) factor 
C                H                   - Hamiltonian Matrix
C                H_temp              - temporary Hamiltonian Matrix
C                N_i                 - Normalization consant for bra
C                N_j                 - Normalization consant for ket
C                H_i                 - Hermite polynomial array for bra
C                H_j                 - Hermite polynomial array for ket
C                Hi                  - Hermite polynomial coefficient for bra
C                Hj                  - Hermite polynomial coefficient for ket
C                L                   - Left limit of integration
C                R                   - Right limit of integraion
C                x                   - Arrary of distance values used for integration
C                hh                  - # Initial value for polynomial portion of the integral
C                y1                  - Array for exponential values for integration
C                y2                  - Array for 1/R^2 values for integration
C                y3                  - Array for the polynomial portion of the integral
C                y                   - Array for entire wave function
C                inte                - running value of integral
C
C            Returns:
C                H - Centrifugal potential Hamiltonian tensor

        interface
          function Hermite(X) RESULT(H)
            INTEGER :: X
            DOUBLE PRECISION :: H(X)
          end function
          function Norm(BETA,V) RESULT (N)
            DOUBLE PRECISION :: BETA, N
            INTEGER :: V
          end function
          function Fac(N) RESULT(P)
            INTEGER :: N
            DOUBLE PRECISION :: P
          end function
          function Ex(N) RESULT(X)
            INTEGER :: N
            DOUBLE PRECISION :: X
          end function
        end interface


        INTEGER :: V, TRAP
        DOUBLE PRECISION :: REQ, BETA

        DOUBLE PRECISION :: L, R, X(TRAP)
        DOUBLE PRECISION :: Y1(TRAP), Y2(TRAP), Y3(TRAP), Y(TRAP)

        DOUBLE PRECISION :: HERMI(0:V), HERMII(0:V)
        DOUBLE PRECISION :: NORMI, NORMII
        DOUBLE PRECISION :: INTE

        DOUBLE PRECISION :: H(0:V,0:V)

        DOUBLE PRECISION :: HBAR, ANG2M
        PARAMETER (HBAR = 1.054571817E-34)
        PARAMETER (ANG2M = 1E-10)

        
        H = 0.d0

        NORMI = 0.d0
        NORMJ = 0.d0

        L = -0.9 * REQ
        R = -L

        DO I=1,TRAP
          X(I) = L + (R-L) * (I-1) / (TRAP-1)
        ENDDO

        Y1 = EXP(-(BETA*X)**2)
        Y2 = (X + REQ)**2

        DO NV=0,V
          HERMI = Hermite(NV)
          NORMI = Norm(BETA, NV)

          HERMJ = 0

          DO NVV=NV,V
            HERMII = Hermite(NVV)
            NORMII = Norm(BETA, NVV)

            Y3 = 0.d0

            DO I=0,NV
              IF (HERMI(I) .NE. 0) THEN
                DO II=0,NVV
                  IF (HERMII(II) .NE. 0) THEN
                    Y3 = Y3 + HERMI(I) * HERMII(II) *
     1                   (BETA*X)**(I+II)
                  ENDIF
                ENDDO
              ENDIF
            ENDDO

            INTE = 0.d0

            Y = (Y1 / Y2) * Y3

            DO I=2,TRAP
              INTE = INTE + (X(I) - X(I-1)) * 0.5 * (Y(I) + Y(I-1))
            ENDDO

            H(NV,NVV) = INTE * NORMI * NORMII
            H(NVV,NV) = INTE * NORMI * NORMII

          ENDDO
        ENDDO

        OPEN (unit=12, file='cent.tmp', status='REPLACE')
        WRITE (12,*) H
        CLOSE (unit=12)


        END

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

        DOUBLE PRECISION function Norm(BETA,V) RESULT (N)

C
C        Function to calculate the normalization constant 
C        for a harmonic oscillator wave fucntion of order V
C
C            Variables:
C                V      - Vibrational quantum number
C                BETA   - Beta value for the diatomic molecule
C
C            Returns:
C                N      - Normalization value

        interface
          function Fac(N) RESULT(P)
            INTEGER :: N
            DOUBLE PRECISION :: P
          end function
          function Ex(N) RESULT(X)
            INTEGER :: N
            DOUBLE PRECISION :: X
          end function
        end interface

        DOUBLE PRECISION :: BETA
        INTEGER :: V

        DOUBLE PRECISION :: PI
        PARAMETER (PI = 4 * ATAN(1.d0))

        N = (1.d0 / SQRT(SQRT(PI))) * SQRT(BETA / (Ex(V) * Fac(V)))

        END function
        
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

        DOUBLE PRECISION function Hermite(X) RESULT(H)

C       Function used to calculate a Hermite polynomial of arbitrary order. 
C
C            Variables:
C                X     - Order of the Hermite polynomial
C                HN1   - Array of the next lowest Hermite polynomial
C                HN    - Array of the current Hermite polynomial
C                HNP1  - Array of the next highest Hermite polynomial
C            
C            Returns:
C                HNP1  - An array where the index and value correspond to the 
C                        exponent and the coefficient, respectively.
C
C            Examples:
C                H0(x) =  1                      --> [1]
C                H1(x) =  0 + 2x                 --> [ 0, 2]
C                H2(x) = -2 +  0x + 4x^2         --> [-2, 0, 4]
C                H3(x) =  0 - 12x + 0x^2 + 8x^3  --> [ 0,-12, 0, 8]
C

        INTEGER :: X, I, D
        DOUBLE PRECISION :: H
        DOUBLE PRECISION :: HN1(0:X), DHN1(0:X)
        
        DIMENSION :: H(0:X)

        HN1  = 0.d0
        DHN1 = 0.d0
        H    = 0.d0

        IF (X .EQ. 0) THEN
          H(0) = 1.d0
        ELSE IF (X .EQ. 1) THEN
          H(0) = 0.d0
          H(1) = 2.d0
        ELSE
          DO I=2,X
            IF (I .EQ. 2) THEN
              HN1(0) = 0.d0
              HN1(1) = 2.d0

              DHN1(0) = 2.d0
            ELSE
              HN1 = H
              
              DO D=0,X-1
                DHN1(D) = HN1(D+1) * (D+1)
              ENDDO

            ENDIF

            H = 0
            H(1:X) = 2 * HN1(0:X-1)
            H = H - DHN1

            HN1 = H

          ENDDO
        ENDIF

        END function
          

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC


CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC


CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

        DOUBLE PRECISION function Ex(N) RESULT(X)
C      
C       Calculates the factorial of n
C
        INTEGER :: N, I

        IF (N .EQ. 0) THEN
          X = 1.d0
        ELSE
          X = 2.d0
          DO I=1,N-1
            X = X*2.d0
          ENDDO
        ENDIF

        END function


CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

        DOUBLE PRECISION function Fac(N) RESULT(P)
C      
C       Calculates the factorial of n
C
        INTEGER :: N, I

        P = 1.d0

        DO I=1,N
            P = P*I
        ENDDO

        END function

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC


        

C END FILE HamilF.f90

C FILE: HamilF.f90

        subroutine Excitations(VAL,VEC,V,MAXV,J,TDM)

C
C       Subroutine used to evaluate all rovibrational excitations
C
C       Variables:
C               EXC - Matrix to store all excitation data
C                     Quantum numbers (4)
C                     Energy levels (3)
C                     Intensity of transtion (3)
C                TDM             - Transition dipole moment matrix
C                VF              - Final vibrational state
C                VI              - Initial vibrational state
C                JF              - Final rotational state
C                JI              - Initial rotational state
C                EF              - Final energy level
C                EI              - Initial energy level
C                E               - Transition energy
C                T               - Transition dipole moment element
C                VEC             - Eigenvectors
C                VAL             - Eigenvalues
C                S               - Honl-London factor
C                F               - f-value for the transition
C                A               - Einstein A-coefficient
C            Returns:
C                excitations_mat


        DOUBLE PRECISION :: VAL(:,:),VEC(:,:,:), TDM(:,:)
        INTEGER :: V, J, MAXV

        INTEGER :: VF, VI, JF, JI
        DOUBLE PRECISION :: EF, EI, E, T(1), Tt(1,MAXV+1)

        DOUBLE PRECISION :: EXC(MAXV*MAXV*3*J,10)

        INTEGER :: S
        DOUBLE PRECISION :: F, A

        DOUBLE PRECISION :: E0, ME, PI, HBAR, CM2J, C, D2CM, D2AU, EC

        PARAMETER (E0 = 8.85418781762039E-12) ! Permittivity of Free Space (C^2  N^-1  m^-2)
        PARAMETER (ME = 9.10938356e-31)       ! Electron Rest Mass (kg)
        PARAMETER (EC = 1.602176634E-19)       ! ELectron Charge (C)
        PARAMETER (PI = 4.d0 * ATAN(1.d0))    ! Pi
        PARAMETER (H = 6.62607015)        ! Plancks constant
        PARAMETER (He = 1E-34)
        PARAMETER (HBAR = 1.054571817E-34)    ! Reduced Plancks constant
        PARAMETER (CM2J = 1.98630E-23)        ! Conversion factor between wavenumbers and Joules
        PARAMETER (C = 2.99792458E8)          ! Speed of light (m/s)
        PARAMETER (D2CM = 3.33564E-30)        ! Conversion factor between Debye and coulomb-meters
        PARAMETER (D2AU = 1 / 2.541)          ! Conversion factor between Debye and atomic units

        EXC = 0.d0

        TDM = TDM / D2AU

        DO VF=0,V
          DO VI=0,V
            DO JF=0,J
              DO JI=MAX(0,JF-1),MIN(JF+1,J)

                EF = VAL(JF+1,VF+1)
                EI = VAL(JI+1,VI+1)

                E = (EI - EF) * CM2J

                IF (ABS(E) .GT. 0.d0) THEN

                  Tt(1,:) = MATMUL(TDM, VEC(JI+1,VI+1,:))
                  T(:)    = MATMUL(Tt, VEC(JF+1,VF+1,:)) * D2CM

                  IF (E .LT. 0.d0) THEN

                    IF (JF .EQ. JI) THEN
                      S = 1
                    ELSE IF (JF .LT. JI) THEN
                      S = JF + 1
                    ELSE
                      S = JF
                    ENDIF

                    F = (8. * PI*PI * ME * E * S * T(1)*T(1))  / 
     1                  (3. * H*H * EC*EC * (2 * JF + 1))
                    F = F / He
                    F = F / He
                    A = (64. * PI**4 * E*E*E * S * T(1)*T(1)) / 
     1                  (3. * H*H*H*H * C*C*C * (2 * JF + 1)  * 
     2                  (4. * PI * E0))
                    A = A / He
                    A = A / He
                    A = A / He
                    A = A / He

                  ELSE
                    F = 0.d0
                    A = 0.d0

                  ENDIF

                  OPEN (unit=12, file='exc.tmp', position='APPEND')
                  WRITE (12,*) VF,JF,VI,JI, EF,EI,E/CM2J, T(1)/D2CM,F,A
                  CLOSE (unit=12)

                ENDIF
              ENDDO
            ENDDO
          ENDDO
        ENDDO

        END

######################################################
# Parametres pouvant etre specifies a l'appel du make
######################################################

# Nombre de processeurs
NP=8

######################################################

# Executable par defaut pour les regles generiques
EXEC=jacobi

EXECS=jacobi gauss_seidel


######################################################
# Diverses constantes et regles.
######################################################
# Pour que les erreurs durant les tests ne fassent pas avorter l'execution.
.IGNORE:

USAGER = $(USER)

MPICH=/home/manuel/openMPI/bin
MAKE = make
RM = rm -f
CC = $(MPICH)/mpicc
RUN = $(MPICH)/mpirun
EXIT = $(MPICH)/mpdallexit
CFLAGS = -Wall -std=c99

compile-jacobi: jacobi

jacobi: jacobi.c
	$(CC) $(CFLAGS) -o jacobi jacobi.c helpers.c -lm

.copies_effectuees1: jacobi
	@cp jacobi /home/$(USAGER)
	@./cp-sur-blades.sh /home/$(USAGER)/jacobi
	@touch .copies_effectuees1

run-jacobi: .copies_effectuees1 .mpdboot compile
	@echo "*** On lance l'execution"
	@(cd /home/$(USAGER); $(RUN) -np $(NP) jacobi 8 200)

compile-jacobi-opti: jacobi-opti

jacobi-opti: jacobi-opti.c
	$(CC) $(CFLAGS) -o jacobi_opti jacobi-opti.c helpers.c -lm

.copies_effectuees2: jacobi-opti
	@cp jacobi /home/$(USAGER)
	@./cp-sur-blades.sh /home/$(USAGER)/jacobi_opti
	@touch .copies_effectuees1

run-jacobi-opti: .copies_effectuees1 .mpdboot compile
	@echo "*** On lance l'execution"
	@(cd /home/$(USAGER); $(RUN) -np $(NP) jacobi_opti 8 200)

compile-gauss_seidel: gauss_seidel

gauss_seidel: gauss-seidel.c
	$(CC) $(CFLAGS) -o gauss_seidel gauss-seidel.c helpers.c -lm

.copies_effectuees3: gauss_seidel
	@cp gauss_seidel /home/$(USAGER)
	@./cp-sur-blades.sh /home/$(USAGER)/gauss_seidel
	@touch .copies_effectuees2

run-gauss_seidel: .copies_effectuees3 .mpdboot compile
	@echo "*** On lance l'execution"
	@(cd /home/$(USAGER); $(RUN) -np $(NP) gauss_seidel 8 200)


######################################################
######################################################

######################################################
# Regle par defaut (compilation) et autres regles 
# (pour executable generique).
######################################################
compile: $(EXEC)

.mpdboot:
	@($(RM) .mpdboot*)
#	@($(EXIT) >& /dev/null)
	@(cp mpd.hosts /home/$(USAGER))
	@./activer-mpdboot.sh $(NP) mpd.hosts

.mpdboot1:
	@($(RM) .mpdboot*)
	@($(EXIT) >/dev/null)
	@(cp unemachine.txt /home/$(USAGER))
	@./activer-mpdboot.sh 1 unemachine.txt 1

.mpdboot2:
	@($(RM) .mpdboot*)
	@($(EXIT) >/dev/null)
	@(cp deuxmachines.txt /home/$(USAGER))
	@./activer-mpdboot.sh 2 deuxmachines.txt 2

.copies_effectuees: $(EXEC)
	@cp $(EXEC) /home/$(USAGER)
	@./cp-sur-blades.sh /home/$(USAGER)/$(EXEC)
	@touch .copies_effectuees

run: .copies_effectuees .mpdboot
	@(cd /home/$(USAGER); $(RUN) -np $(NP) $(EXEC) $(ARG1) $(ARG2))

debug_un: .copies_effectuees .mpdboot1
	@(cd /home/$(USAGER); $(RUN) -np $(NP) $(EXEC) $(ARG1) $(ARG2))

debug_deux: .copies_effectuees .mpdboot2
	@(cd /home/$(USAGER); $(RUN) -np $(NP) $(EXEC) $(ARG1) $(ARG2))


########################################
# Reconstruction totale et/ou nettoyage.
########################################

TMPFILES = $(EXECS)\
           TAGS \
           *~ *.o \

all: compile .copies_effectuees run

public: cleanxtra
	~/cmd/creer-version-etudiant.sh Labo1

clean:
	@$(RM) $(TMPFILES)
	@$(RM) /home/$(USAGER)/unemachine.txt
	@$(RM) /home/$(USAGER)/deuxmachines.txt
	@$(RM) *~
	@$(RM) .mpdboot* .copies_effectuees*

exit: clean
	@($(EXIT) >/dev/null)

cleanxtra: clean exit
	@$(RM) -r jacobi


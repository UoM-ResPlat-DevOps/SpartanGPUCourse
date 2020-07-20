CC=pgcc
ifeq ($(VIS),1)
	INC=-DVIS -I../../include/
endif
ACC_FLAGS=-fast -Minfo=all -acc
CFLAGS=$(ACC_FLAGS) -DROWS=2000 -DCOLS=2000 $(INC)
EXE=heat_eq_acc_v2
all: heat_eq_acc_v2

heat_eq_acc_v2: heat_eq.o
	$(CC) $(CFLAGS) heat_eq.o -o $(EXE)

%.o: %.c
		$(CC) -c -o $@ $< $(CFLAGS)

clean:
		rm *.o $(EXE)
cleanall:
		rm *.o *.out *.vtk $(EXE)

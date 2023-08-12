CFLAGS = -Wall -Wshadow -O3 -g -march=native
LDLIBS = -lm

all: sensor small_sensor node

sigmoid: CFLAGS += -Dgenann_act=genann_act_sigmoid_cached
sigmoid: all

threshold: CFLAGS += -Dgenann_act=genann_act_threshold
threshold: all

linear: CFLAGS += -Dgenann_act=genann_act_linear
linear: all

sensor: sensor.o genann.o

small_sensor: small_sensor.o genann.o

node: node.o genann.o

clean:
	$(RM) *.o
	$(RM) sensor small_sensor node *.exe
	$(RM) persist.txt

.PHONY: sigmoid threshold linear clean

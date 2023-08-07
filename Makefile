CFLAGS = -Wall -Wshadow -O3 -g -march=native
LDLIBS = -lm

all: check example1 example2 example3 example4 sensor small_sensor node

sigmoid: CFLAGS += -Dgenann_act=genann_act_sigmoid_cached
sigmoid: all

threshold: CFLAGS += -Dgenann_act=genann_act_threshold
threshold: all

linear: CFLAGS += -Dgenann_act=genann_act_linear
linear: all

test: test.o genann.o

check: test
	./$^

example1: example1.o genann.o

example2: example2.o genann.o

example3: example3.o genann.o

example4: example4.o genann.o

sensor: sensor.o genann.o

node: node.o genann.o

small_sensor: small_sensor.o genann.o


clean:
	$(RM) *.o
	$(RM) test example1 example2 example3 example4 sensor small_sensor node *.exe
	$(RM) persist.txt

.PHONY: sigmoid threshold linear clean

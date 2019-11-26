#!/usr/bin/env python3
from cacho import *
import time


def main():
    time_rolls(Cup(), None, 100000)

def commas(num):
	return "{:,}".format(num)

def time_rolls(cup1, cup2, num_rolls):
	print('Timing', commas(num_rolls), 'rolls...')

	start1 = time.time()
	for i in range(num_rolls):
		cup1.shake()
	end1 = time.time()

	print('Cup 1:', commas(round(num_rolls / (end1-start1))), 'r/s')

	if cup2 is None:
		return

	start2 = time.time()
	for i in range(num_rolls):
		cup2.shake()
	end2 = time.time()

	print('Cup 2:', commas(round(num_rolls / (end2-start2))), 'r/s')


if __name__ == "__main__":
    main()
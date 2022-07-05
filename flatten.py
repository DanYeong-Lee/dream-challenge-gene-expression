import argparse
import json
import numpy as np
from scipy import stats

def gaussian_to_uniform(x):
    return stats.norm.cdf(x, loc=x.mean(), scale=x.std()) * 3 - 1.5

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input')
	parser.add_argument('-o', '--output')
	args = parser.parse_args()

	with open(args.input, 'r') as inFile, open(args.output, 'w') as outFile:
		orig = json.loads(inFile.read())
		predictions = np.array(list(orig.values()))
		flattened = dict(zip(orig.keys(), gaussian_to_uniform(predictions)))
		print(json.dumps(flattened), file=outFile)
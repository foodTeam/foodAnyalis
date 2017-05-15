# tool function
def readFile(filePath):
    file = open(filePath)
    for line in file:
        lineStr = line.split(',')
        meanValue = lineStr[0].split(':')[1]
        nEstimator = lineStr[2].split(':')[2].split('}')[0]
        print(meanValue + " " + nEstimator)
    return meanValue,nEstimator

if __name__ == '__main__':
    filePath = "score.txt"
    [meanValue,nEstimator] = readFile(filePath)
    

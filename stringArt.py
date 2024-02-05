import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import random
  
def greyScale(data, mode=1):
    if data.max() > 1:
        data = data / 255
    if mode == 0:
        return np.array(np.sqrt(0.299 * data[:, :, 0]**2 + 0.587 * data[:, :, 1]**2 + 0.114 * data[:, :, 2]**2))
    elif mode == 1:
        return np.array(0.299 * data[:, :, 0] + 0.587 * data[:, :, 1] + 0.114 * data[:, :, 2])
    elif mode == 2:
        return np.array(0.2126 * data[:, :, 0] + 0.7152 * data[:, :, 1] + 0.0722 * data[:, :, 2])
    
def pixelate(data, displayWidth):
    # Takes in an image data and compress it to keep every nth pixel of an axis, all values should be normalized
    height, width = data.shape[0:2]
    pixelateFactor = width // displayWidth

    pixelated = data[::pixelateFactor, ::pixelateFactor]
    return pixelated

def dither(img):
    dithered = img.copy()
    height, width = img.shape
    if dithered.max() > 1.0:
        dithered = dithered / 255

    for y in range(height-1):
        for x in range(width-1):
            old = dithered[y, x]
            new = round(old)
            # print(new)
            dithered[y, x] = new
            quantError = old - new
            dithered[y+1][x] += quantError * 7 / 16
            dithered[y+1][x-1] += quantError * 3 / 16
            dithered[y+1][x] += quantError * 5 / 16
            dithered[y+1][x+1] += quantError * 1 / 16

    dithered = np.clip(dithered, 0, 1)
    return dithered

def plotLineLow(x0, y0, x1, y1):
    dx = x1 - x0
    dy = y1 - y0

    yi = 1
    if dy < 0:
        yi = -1
        dy = -dy

    D = 2*dy - dx
    y = y0

    noPointsDrawn = dx + 1
    # lineCoors = np.empty((2, noPointsDrawn), dtype=np.uint16)
    xs = np.empty(noPointsDrawn, dtype=np.uint)
    ys = np.empty(noPointsDrawn, dtype=np.uint)

    for i, x in enumerate(range(x0, x1+1)):
        xs[i] = x
        ys[i] = y
        if D > 0:
            y += yi
            D = D + (2 * (dy - dx))
        else:
            D += 2*dy
    
    return xs, ys

def plotLineHigh(x0, y0, x1, y1):
    dx = x1 - x0
    dy = y1 - y0

    xi = 1
    if dx < 0:
        xi = -1
        dx = -dx

    D = 2*dx - dy
    x = x0

    noPointsDrawn = dy + 1
    xs = np.empty(noPointsDrawn, dtype=np.uint)
    ys = np.empty(noPointsDrawn, dtype=np.uint)

    for i, y in enumerate(range(y0, y1+1)):
        xs[i] = x
        ys[i] = y
        if D > 0:
            x += xi
            D = D + (2 * (dx - dy))
        else:
            D += 2*dx

    return xs, ys

def plotLine(x0, y0, x1, y1):
    if abs(y1 - y0) < abs(x1 - x0):
        if x0 > x1:
            return plotLineLow(x1, y1, x0, y0)
        else:
            return plotLineLow(x0, y0, x1, y1)
    else:
        if y0 > y1:
            return plotLineHigh(x1, y1, x0, y0)
        else:
            return plotLineHigh(x0, y0, x1, y1)

class Pins:
    def __init__(self, noOfPins, dimension):
        self.noOfPins = noOfPins

        self.pixelDensity = dimension // 2
        self.dimension = dimension

        self.pinsPositions = []
        
        for i in range(noOfPins):
            theta = (2 * np.pi / noOfPins) * i
            point = (np.cos(theta), np.sin(theta))
            self.pinsPositions.append(point)

        # pinsPerRow = int(np.sqrt(noOfPins))
        # self.pinsPositions = [(x, y) for x in np.linspace(-1, 1, pinsPerRow) for y in np.linspace(-1, 1, pinsPerRow)]
        # self.noOfPins = len(self.pinsPositions)

        # print(self.noOfPins)

        self.allStrings = {}

        # allStrings = np.zeros((self.dimension+1, self.dimension+1), dtype=int)

        for i in range(self.noOfPins-1):
            # print(f'Start pin {i}')
            x0, y0 = self.pinsPositions[i]
            x0_i = int((x0 + 1) * self.pixelDensity)
            y0_i = int((-y0 + 1) * self.pixelDensity)

            for j in range(i+1, self.noOfPins):
                xn, yn = self.pinsPositions[j]
                xn_i = int((xn + 1) * self.pixelDensity)
                yn_i = int((-yn + 1) * self.pixelDensity)

                lineCoors = plotLine(x0_i, y0_i, xn_i, yn_i)

                self.allStrings[(i, j)] = lineCoors
                # xs, ys = lineCoors
                # allStrings[xs, ys] = 1
                # plt.imshow(allStrings)
                # plt.show()

        self.resultStrings = np.zeros((self.dimension+1, self.dimension+1), dtype=float)
        # input()
        # input()

    def stringArt(self, image, maxIterations, linesPerIteration, opacity):
        allPointsIndex = list(range(self.noOfPins))
        startPoint = allPointsIndex.pop()

        # currMask = np.empty((self.dimension+1, self.dimension+1), dtype=bool)

        for i in range(maxIterations):
            # print(f'Iteration {i}')
            # Finds the line that cover the most ammount of darkness in the picture
            biggestAvgDark = 0.0
            randEndPoints = random.sample(allPointsIndex, k=linesPerIteration)
            for randEndPoint in randEndPoints:
                # imageCopy = image.copy()
                # Do I need to exclude i?
                # Answer: YES
                try:
                    xs, ys = self.allStrings[(startPoint, randEndPoint)]
                except KeyError:
                    xs, ys = self.allStrings[(randEndPoint, startPoint)]
                 
                avgDark = image[ys, xs].sum() / len(xs)

                if avgDark > biggestAvgDark:
                    biggestAvgDark = avgDark
                    bestXs = xs
                    bestYs = ys
                    nextStartPoint = randEndPoint
                
                
                # imageCopy[xs, ys] = 1
                # print(f'Avg Darkness: {avgDark}')
                # plt.imshow(imageCopy, cmap='binary')
                # plt.show()

                # currMask.fill(0)
                
            if biggestAvgDark < 0.0:
                print('Stop at string', i)
                break
            
            self.resultStrings[bestYs, bestXs] += opacity
            image[bestYs, bestXs] -= opacity
            # image = np.clip(image, 0, 1)

            # print(f'Settle on pin {startPoint} to {nextStartPoint} with biggest avgDark: {biggestAvgDark}')
            # plt.imshow(image, cmap='binary')
            # plt.colorbar()
            # plt.show()

            allPointsIndex.append(startPoint)
            startPoint = nextStartPoint
            allPointsIndex.remove(nextStartPoint)
        
        self.resultStrings = np.clip(self.resultStrings, 0, 1)

    def draw(self, name):
        plt.imshow(self.resultStrings, cmap='binary')
        plt.imsave(r"C:\Users\PC\Downloads" + rf"\{name}str.png", self.resultStrings, cmap='binary', vmin=0.0, vmax=1.0)
        plt.show()

def main():
    # path = r"C:\Users\PC\Downloads\Niko - Copy.jpg"
    # path = r"C:\Users\PC\Downloads\animegirl.png"
    path = r"C:\Users\PC\Downloads\abraham - Copy.png"
    # path = r"C:\Users\PC\Downloads\Puro.jpg"
    # path = r"C:\Users\PC\Pictures\Various Artists's Drawings\niko bindows.png"
    # path = r"C:\Users\PC\Downloads\animeGirl3.png"
    # path = r"C:\Users\PC\Downloads\feedme.png"
    # path = r"C:\Users\PC\Downloads\3915930.png"
    # path = r"C:\Users\PC\Downloads\napoleon.png"
    # path = r"C:\Users\PC\Downloads\93e2379.jpg"
    # path = r"C:\Users\PC\Downloads\Albert-Einstein.jpg"
    # path = r"C:\Users\PC\Downloads\cat4.png"
    img = Image.open(path)
    data = np.array(img)

    name = path.split('Downloads\\')[1].split('.png')[0]

    imgLength = data.shape[0]
    # mode 0 sucks, mode 1 works best
    greyScaled = 1 - greyScale(data, mode=1)
    pixelated = pixelate(greyScaled, imgLength)
    # dithered = dither(pixelated)
    # No significant difference found

    # pixelated = np.zeros((700, 700), dtype=float)
    # pixelated[0:300, 0:700] = 1

    if imgLength % 2 == 0:
        pixelated = np.pad(pixelated, ((0, 1), (0, 1)), mode='constant', constant_values=0)

    dimension = pixelated.shape[0]

    plt.imshow(pixelated, cmap='binary')
    plt.colorbar()
    plt.show()

    pins = Pins(600, dimension)
    pins.stringArt(pixelated, 18000, 400, 0.1)
    pins.draw(name=name)

main()
        
 
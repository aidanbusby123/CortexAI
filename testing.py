import numpy as np

import TM
import Encoding


np.set_printoptions(threshold=np.inf)

def main():
    spatial_dims = (32, 32)
    dims = (32, 32, 8)
    time_dim = (2,)

    


    network = TM.TMLayer(dims, time_dim)

    encoding = Encoding.SimpleEncoder(spatial_dims, 10, 96)



    print("Enter input string")

    inputstr = input()

    

    for char in inputstr:
        val = ord(char) - 32
        inputEncoding = encoding.encode(val)

        network.update(inputEncoding)

        #print(inputEncoding)


if __name__ == "__main__":
    main()
import numpy as np
import pygame
import sys


import TM
import Encoding


np.set_printoptions(threshold=np.inf)

def start_visualization(get_data_func, cell_size=20):
    """
    model: Your TMLayer instance
    get_data_func: A lambda or function that returns model.tm_a
    """
    # Initialize Pygame
    pygame.init()
    
    # Get dimensions from the data
    sample_data = get_data_func()
    rows, cols = sample_data.shape
    
    width = cols * cell_size
    height = rows * cell_size
    
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("DMN / TMLayer Activity Visualization")
    clock = pygame.time.Clock()


    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        # 1. Step the model (This calls your update logic)
        # Assuming you have some 'dummy_input' or real stream
        # model.update(dummy_input) 

        # 2. Get the current activations
    data = get_data_func()
        
    # 3. Normalize for visualization (0-255)
    # We clip to ensure NaNs or INF don't crash the renderer
    data = np.nan_to_num(data)
    normalized_data = np.clip(data * 255, 0, 255).astype(np.uint8)
     # 4. Draw the Grid
    screen.fill((20, 20, 25)) # Dark background
    
    for r in range(rows):
            for c in range(cols):
                val = normalized_data[r, c]
                # Map intensity to a "DMN Blue/Cyan" or "Fire" color scheme
                # Using (val, val//2, 255) creates a nice neuro-electric blue
                color = (val, min(255, val + 50), min(255, val + 150)) if val > 0 else (10, 10, 15)
                
                pygame.draw.rect(
                    screen, 
                    color, 
                    (c * cell_size, r * cell_size, cell_size - 1, cell_size - 1)
                )

    pygame.display.flip()
    clock.tick(60) # Limit to 30 FPS to save CPU for the DMN math

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
        start_visualization(network.get_tm_activations)
        #print(inputEncoding)


if __name__ == "__main__":
    main()
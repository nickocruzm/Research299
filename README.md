# README

## Training Flow

1. Load an image from a dataset
2. Encode it using the **VAE** encoder. This gives a Latent Tensor $z_{0}$ 
	- If our image was 512x512 RGB, the $z_{0}$ would have the shape (4,64,64)
3. Apply noise to $z_{0}$ using the **noise scheduler**: $x_{t} = \sqrt{\alpha_t}z_{0} + \sqrt{1 - \alpha_{t}}{\epsilon}$ 
4. Feed $x_t$ at timestep $t$, and *text embedding* $c$ into the **U-Net** to predict $\epsilon$
5. Compute Loss
6. Backpropagate (update the U-Net weights)

## Links
Notebook: https://colab.research.google.com/drive/1cMkft2zsIJSDG_yn09G03TMd7qXU6SZh?usp=sharing

AYS EXAMPLE: https://colab.research.google.com/drive/1cIwbbO4HRP1aUQ8WcbQBaT8p3868k7BC?usp=sharing

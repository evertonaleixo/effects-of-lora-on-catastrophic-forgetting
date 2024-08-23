import numpy as np
import matplotlib.pyplot as plt

def create_plot_images(sample_cpu, target):
  fig, axes1 = plt.subplots(3, 3, figsize=(9, 9))

  for j in range(3):
    for k in range(3):
      im = np.transpose(sample_cpu[(j*3+k)], (1, 2, 0))
      im = im + max(abs(im.max()), abs(im.min()))
      im = im / max(abs(im.max()), abs(im.min())) 
      im = (im * 255).astype(int)
      ax = axes1[j][k]
      ax.imshow(im, interpolation='nearest')
      legend_label = f'{target[j*3+k]}'
      color = 'black'
      ax.text(0.5, 1.05, legend_label, color=color, transform=ax.transAxes, ha='center', fontsize='medium')
      ax.axis('off')
      print(legend_label)

  plt.subplots_adjust(bottom=0.1, hspace=0.2)
  plt.show()
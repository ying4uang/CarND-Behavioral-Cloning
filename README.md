


# Behavior Cloning  Lab


Through this lab I have utilized the common ai model with slight modification to add regularization. 

## Network Structure
#### common.ai
I followed the guided provided on forum that [Commonai](https://github.com/commaai/research) would be a good place to start and I used that as a baseline. The original model structure from commonai is pasted below:

```sh
def get_model(time_len=1):
  ch, row, col = 3, 160, 320  # camera format

  model = Sequential()
  model.add(Lambda(lambda x: x/127.5 - 1.,
            input_shape=(ch, row, col),
            output_shape=(ch, row, col)))
  model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
  model.add(ELU())
  model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
  model.add(ELU())
  model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
  model.add(Flatten())
  model.add(Dropout(.2))
  model.add(ELU())
  model.add(Dense(512))
  model.add(Dropout(.5))
  model.add(ELU())
  model.add(Dense(1))

  model.compile(optimizer="adam", loss="mse")

  return model
```


#### Processing the Image
Originally I was using my own records and out of box commonai model. The out of the box commonai did not give me very good results. I was reading from comments on slack and forums that adding flip to the images and l2 regularization. After randomly flipping the images and using the udacity dataset, my car was able to proceed a bit further but still cannot make through the first curve. For next step I will try adding the left and right images as well.





```python

```

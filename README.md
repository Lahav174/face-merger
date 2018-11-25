# face-merger

Download TFD from here: http://vis-www.cs.umass.edu/lfw/

Here is the google cloud vm: https://console.cloud.google.com/compute/instances?project=face-merger&organizationId=819335046878&duration=PT1H

## Updates

I am trying to get the adverserial autoencoder to produce non-blurry images. The network structure is more complex now, and I am just playing with parameters. Also FYI Carlo, I created an gcloud vm for us to use. Check your email, I set you as co-owner -Lahav

## TODO
1. ~~Replace CIFAR dataset in variational_autoencoder.py with TFD dataset. This involves just replacing the code with what's in custom_dataset.py. However, we may need to increase the depth of the neural network to account for the increased complexity if we still want the network to reconstruct the images well.~~ In order for the code to run, you have to download the Toronto Face Dataset (TFD) from the link below.

2. ~~Use the adverserial_autoencoder.py as a template for including an adverserial net in parallel with our variational autoencoder. Mess around with weighting the loss functions differently. This should also be a quick implementation.~~

3. Implement an alternating scheme, where after each batch, the loss function switches between reconstruction loss, and discriminative loss. Remember that with the discriminative loss, we are propagating two embeddings through the network, then averaging them at the halfway point, then propagating them through and checking the discriminative loss.

### Additional task
Find another dataset of images, other than faces, which we can use

## Citations

Based dataset class off of the one here: https://github.com/utkuozbulak/pytorch-custom-dataset-examples

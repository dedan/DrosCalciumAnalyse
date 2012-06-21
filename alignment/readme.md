
# General Comments

We realized that the alignment of the images that we get from the experimentalists is not perfect. They use some GUI that all biologists use, I forgot the name. This folder here contains experiments of how the alignment of images could be improved.

The first (and currently only attempt) is to use the algorithm from [here](http://xanthippi.ceid.upatras.gr/people/evangelidis/ecc/).

I did not manage to get the C version of the algorithm running and currently used its matlab implementation in the script `test.m`

Description of all files in this folder:

* test.m: matlab script to run the algorithm on previously saved .pgm images
* ts_to_images.py: creates what `test.m` needs, the images from TimeSeries objects
* align.py: is a alignment script that jan gave me and that I didn't use yet.

To create videos from the images I used the following command:

    ffmpeg -r 10 -i test%03d.pgm ../janalign.mp4


### some more comments

which projection?

	* homography is too complex, leads to strange drift
	* translation is not enough
	* ==> chose affine projection

number of iterations of the algorithm?

	* there is this NoI parameter. I tried different values (5, 40, 100)
	* seems like there is no big difference

is there a better evaluation than just looking at the videos?

	* average difference between to following images could be used
	* first idea is to use this fitness function to see whether the NoI param really doesn't make a difference
	* results for NoI 5=4599164.0, 40=4611763.0, 80=4611763.0
	* seems to make no difference and seems to converge extremely stable. So I'll use 5 in the future



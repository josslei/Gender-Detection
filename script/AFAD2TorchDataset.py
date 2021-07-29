import os
from random import randint
import sys

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print()
        print('[1] Please specify the source dir!')
        print('[2] Please specify the destination dir!')
        print('[3] Please specify the number of images for testing!')
        exit()
    # root dir
    top = sys.argv[1]
    print('...')
    print('Reading data under "' + top + '"')
    print('....\n.....\n......\n')
    # list of image file names (relative full path)
    male_img = []
    female_img = []
    # list all directories
    dirs = []
    dirs += [os.path.join(top, x) for x in os.listdir(top)
             if os.path.isdir(os.path.join(top, x))]
    # dirs.sort()

    # get all images' paths
    for i in dirs:
        # male
        male_img += [os.path.join(i, '111', x) for x in os.listdir(os.path.join(i, '111'))]
        # female
        female_img += [os.path.join(i, '112', x) for x in os.listdir(os.path.join(i, '112'))]
    # print dataset information
    print('            Number of  male  images: ', len(male_img))
    print('            Number of female images: ', len(female_img))

    # copy images to create
    dest      =     sys.argv[2]
    num_test  = int(sys.argv[3])

    # test sets
    test_male_img = []
    test_female_img = []
    num_test_male_img = int(num_test / 2)
    num_test_female_img = num_test - num_test_male_img
    # generate male test img
    for i in range(num_test_male_img):
        index = randint(0, len(male_img) - 1)
        test_male_img += [male_img[index]]  # append &
        male_img.pop(index)                 # remove
    # generate female test img
    for i in range(num_test_female_img):
        index = randint(0, len(female_img) - 1)
        test_female_img += [female_img[index]]  # append &
        female_img.pop(index)

    print()
    print('  New dataset will be created under: ', dest)
    print('          Amount of training images: ', len(male_img) + len(female_img))
    print('                                     (Male = ', len(male_img), ', Female = ', len(female_img), ')')
    print('           Amount of testing images: ', num_test)
    print('                                     (Male = ', num_test_male_img, ', Female = ', num_test_female_img, ')')
    print()

    # telling a lie ^v^
    # seems like i haven't chosen these images yet
    print('Testing data will be randomly chosen from the original dataset,')
    print('Images that appear in the testing set will not appear in the training set.')
    print('Basically, male : female = 1 : 1\n')
    if input('Are you sure to continue this process? [y/n] ') != 'y':
        exit()

    print()
    print('Processing...')
    # exit()

    '''
    Training set
    '''
    # make dirs
    os.system('mkdir -p ' + dest)
    if 256 == os.system('mkdir -p ' + os.path.join(dest, 'train/male/')):   # error code
        os.system('rm -f ' + os.path.join(dest, 'train/male'))
        os.system('mkdir -p ' + os.path.join(dest, 'train/male/'))
    if 256 == os.system('mkdir -p ' + os.path.join(dest, 'train/female/')): # error code
        os.system('rm -f ' + os.path.join(dest, 'train/female'))
        os.system('mkdir -p ' + os.path.join(dest, 'train/female/'))
    # male
    for m in male_img:
        os.system('cp ' + m + ' ' + os.path.join(dest, 'train/male'))   # male
    print('Train-male done.')
    # female
    for f in female_img:
        os.system('cp ' + f + ' ' + os.path.join(dest, 'train/female')) # female
    print('Train-female done.')
    
    '''
    Testing set
    '''
    # make dirs
    if 256 == os.system('mkdir -p ' + os.path.join(dest, 'test/male/')): # error code
        os.system('rm -f ' + os.path.join(dest, 'test/male'))
        os.system('mkdir -p ' + os.path.join(dest, 'test/male/'))
    if 256 == os.system('mkdir -p ' + os.path.join(dest, 'test/female/')): # error code
        os.system('rm -f ' + os.path.join(dest, 'test/female'))
        os.system('mkdir -p ' + os.path.join(dest, 'test/female/'))
    # male
    for m in test_male_img:
        os.system('cp ' + m + ' ' + os.path.join(dest, 'test/male'))    # male
    print('Test-male done.')
    # female
    for f in test_female_img:
        os.system('cp ' + f + ' ' + os.path.join(dest, 'test/female'))  # female
    print('Test-female done.')

    print('done.')

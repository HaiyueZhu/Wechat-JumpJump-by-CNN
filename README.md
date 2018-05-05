# Wechat-JumpJump-by-CNN
Using Convolutional Neural Network (CNN) to play Wechat Jump-Jump Game

The idea to using CNN to play Wechat Jump-Jump Game is to mimic our human how the play this game that we see the start point and the target point, and then judge their distance to press the scree. So we can train a CNN model to find the start point's and the target point's position in an image, and then calculate their distance to press using ADB tool.

For training the CNN model, there are some image processing based method to play Wechat Jump-Jump Game, as we found the     following one: https://github.com/wangshub/wechat_jump_game. We used their method to generate the training images by auto run it and saving every step's image with the name as the four coordinates of the starting jump point and ending point (target).

There are three steps for playing the games:
      1, Run GetCNNTrainingImages.py to get the CNN training images
      2, Run CNN_Training.py to creat a CNN (tensorflow) and train it
      3, Play the jump-jump game using the trained CNN

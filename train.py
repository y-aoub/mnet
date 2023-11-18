def crop_and_apply_mask(image_path, mask_path):
    # Load the image
    image = mpimg.imread(image_path)

    # Load the mask
    mat_data = scipy.io.loadmat(mask_path)
    mask = mat_data['mask']

    # Crop the mask to match the image size
    mask_height, mask_width = mask.shape[:2]
    image_height, image_width = image.shape[:2]

    start_col = (mask_width - image_width) // 2
    end_col = start_col + image_width

    cropped_mask = mask[:, start_col:end_col]

    # Apply the mask to the image
    result = cv2.bitwise_and(image, image, mask=cropped_mask)

    # Increase the size of the figure
    plt.figure(figsize=(15, 15))

    # Display the original image, cropped mask, and the result
    plt.subplot(131), plt.imshow(image), plt.title('Original Image')
    plt.subplot(132), plt.imshow(cropped_mask, cmap='gray'), plt.title('Cropped Mask')
    plt.subplot(133), plt.imshow(result), plt.title('Image with Mask Applied')
    plt.show()
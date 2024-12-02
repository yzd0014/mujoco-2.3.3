from PIL import Image, ImageChops

def convert_black_to_transparent(image):
    # Convert black pixels to transparent
    image = image.convert("RGBA")  # Ensure RGBA mode
    datas = image.getdata()

    new_data = []
    for item in datas:
        # Change all black (0, 0, 0) pixels to transparent
        if item[:3] == (0, 0, 0):
            new_data.append((255, 255, 255, 0))  # Set to transparent
        else:
            new_data.append(item)

    image.putdata(new_data)
    return image

def convert_black_to_transparent2(image):
    # Convert black pixels to transparent
    image = image.convert("RGBA")  # Ensure RGBA mode
    datas = image.getdata()

    new_data = []
    for item in datas:
        # Change all black (0, 0, 0) pixels to transparent
        if item[:3] == (0, 0, 0):
            new_data.append((255, 255, 255, 0))  # Set to transparent
        else:
            new_item = (item[0], item[1], item[2], int(0.5*255))
            new_data.append(new_item)

    image.putdata(new_data)
    return image

def create_trailing_effect(image_sequence):
    # Get the size from the first image in the sequence
    width, height = image_sequence[0].size
    # Create a blank transparent image for the base
    base = Image.new("RGBA", (width, height), (0, 0, 0, 0))

    # Apply the trailing effect by stacking images with reduced opacity
    for i, img in enumerate(image_sequence):
        # Convert black to transparent in each frame
        img = convert_black_to_transparent(img)
        # Apply the trail effect by blending with decreasing opacity
        base = Image.alpha_composite(base, img)

    return base


# Load your sequence of images
image_sequence = [Image.open(f"frames/frame_{i:04d}.png") for i in range(0, 20)]  # Adjust filenames as needed

# Create the trailing effect
final_image = create_trailing_effect(image_sequence)

image_sequence2 = [Image.open(f"frames/body_{i:04d}.png") for i in range(0, 3)]  # Adjust filenames as needed
for i, img in enumerate(image_sequence2):
    # Convert black to transparent in each frame
    img = convert_black_to_transparent2(img)
    # Apply the trail effect by blending with decreasing opacity
    final_image = Image.alpha_composite(final_image, img)

# Save or show the result
final_image.save('frames/trailing_effect2.png')
final_image.show()
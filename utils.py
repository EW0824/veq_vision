
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np

def visualize_real_with_cad(real_img, cad_images, scores, best_idx):
    fig, axs = plt.subplots(1, len(cad_images) + 1, figsize=(15, 5))

    axs[0].imshow(cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Real Part")
    axs[0].axis('off')

    for i, (cad_img, score) in enumerate(zip(cad_images, scores)):
        axs[i + 1].imshow(cv2.cvtColor(cad_img, cv2.COLOR_BGR2RGB))
        axs[i + 1].set_title(f"CAD View {i+1}\nScore: {score:.2f}")
        axs[i + 1].axis('off')

        # if i == best_idx:
        #     print("Best View: ", i)
        #     pos = axs[i + 1].get_position()
        #     rect = patches.Rectangle(
        #         (pos.x0, pos.y0), pos.width, pos.height,
        #         linewidth=5, edgecolor='green', facecolor='none', transform=fig.transFigure
        #     )
        #     fig.patches.append(rect)  # Add rectangle to figure


    plt.tight_layout()
    plt.show()


def visualize_keypoints(image, title="Keypoints"):
    sift = cv2.SIFT_create()
    keypoints, _ = sift.detectAndCompute(image, None)
    img_with_kp = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    plt.figure(figsize=(5, 5))
    plt.imshow(cv2.cvtColor(img_with_kp, cv2.COLOR_BGR2RGB))
    plt.title(f"{title} ({len(keypoints)} keypoints)")
    plt.axis('off')
    plt.show()



import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__)))
from NeuralStyleTransfer.model import NeuralStyleTransferModel
from NeuralStyleTransfer.utils import image_loader, image_save

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
		
    project_root = os.path.abspath(os.path.realpath(os.path.join(
        os.path.dirname(os.path.realpath(__file__)))))

    parser.add_argument("--content-path", type=str,
                        default=os.path.join(
                            project_root, "data/images", "content.jpg"),
                        help="Path to the content image.")
    parser.add_argument("--style-path", type=str,
                        default=os.path.join(
                            project_root, "data/images", "style.jpg"),
                        help="Path to the content image.")                  
    parser.add_argument("--save-path", type=str,
                        default=os.path.join(
                            project_root, "data/images", "output.jpg"),
                        help="Path to save output image")
                
    args = parser.parse_args()
    
    style_image = image_loader(args.style_path)
    print("Style Images loaded from disk")
    
    content_image = image_loader(args.content_path)
    print("Conent Images loaded from disk")
   
   
    model = NeuralStyleTransferModel(style_image, content_image)
    print("model initialized")
    print()
    
    output_image = model.style_transfer()
    image_save(output_image, args.save_path)
    print(f"output image saved to {args.save_path}")
 
if __name__ == "__main__":
    main()
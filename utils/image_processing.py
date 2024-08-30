from torchvision import transforms

def preprocess_image(image):
    """Prétraite l'image pour l'inférence avec le modèle."""
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),  # Taille que le modèle attend
        transforms.ToTensor(),          # Conversion en tenseur PyTorch
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalisation
    ])
    return preprocess(image)


from flask import Flask, request, jsonify
import torch
from utils.image_processing import preprocess_image
import torchvision.models as models
from PIL import Image

# Initialisation de l'application Flask
app = Flask(__name__)



# Chargement du modèle pré-entraîné
model_path = "modeleState.pt"
num_classes = 13

model = models.resnet50()
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()


# Liste des classes (par exemple, 13 classes)
classes = ['class1', 'class2', 'class3', 'class4', 'class5', 
           'class6', 'class7', 'class8', 'class9', 'class10', 
           'class11', 'class12', 'class13']

classes =   {   0:  "Castor",
                1:  "Chat",
                2:  "Chien",
                3:  "Coyote",
                4:  "Ecureuil",
                5:  "Lapin",
                6:  "Loup",
                7:  "Lynx",
                8:  "Ours",
                9:  "Puma",
                10: "Rat",
                11: "Raton Laveur",
                12: "Renard"
            }


# Définition du point d'entrée pour traiter l'image
@app.route('/process', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Ouvrir l'image
        image = Image.open(file.stream).convert('RGB')
        
        # Prétraitement de l'image
        processed_image = preprocess_image(image).unsqueeze(0)

        # Faire une prédiction
        with torch.no_grad():
            prediction = model(processed_image)

        probabilities = torch.nn.functional.softmax(prediction[0], dim=0)
        top_prob, top_class = torch.topk(probabilities, k=3)

        resultat1 = classes[top_class[0].item()]
        resultat2 = classes[top_class[1].item()]
        resultat3 = classes[top_class[2].item()]

        probresultat1 = round(top_prob[0].item(), 2)
        probresultat2 = round(top_prob[1].item(), 2)
        probresultat3 = round(top_prob[2].item(), 2)

        predicted_class = classes[resultat1]


        # Retourner le résultat
        return jsonify({'class': predicted_class})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Démarrer l'application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
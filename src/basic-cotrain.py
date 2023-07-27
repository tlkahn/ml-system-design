# [[file:../README.org::*Co-trained][Co-trained:1]]
import torch
import torch.nn as nn
import torchvision.models as models
from torchtext.vocab import GloVe

# Set up image and text feature extraction models
resnet18 = models.resnet18(pretrained=True)
resnet18 = nn.Sequential(*list(resnet18.children())[:-1])  # Remove classification layer
glove = GloVe(name="6B", dim=300)


# Co-trained embedding model
class CoTrainedEmbedding(nn.Module):
    def __init__(self, text_dim, image_dim, embedding_dim):
        super(CoTrainedEmbedding, self).__init__()
        self.text_fc = nn.Linear(text_dim, embedding_dim)
        self.image_fc = nn.Linear(image_dim, embedding_dim)

    def forward(self, text, image):
        text_embed = self.text_fc(text)
        image_embed = self.image_fc(image)
        return text_embed, image_embed


# Model parameters
text_dim = 300  # GloVe 300-dimensional embedding
image_dim = 512  # ResNet18 final feature map size
embedding_dim = 128

# Initialize the co-trained embedding model
model = CoTrainedEmbedding(text_dim, image_dim, embedding_dim)

# Example data
text_data = "This is a sample text."
image_data = torch.randn(1, 3, 224, 224)  # Random 224x224 image

# Extract text and image features
text_features = glove.get_vecs_by_tokens(text_data.split())
image_features = resnet18(image_data).squeeze()

# Forward pass through the co-trained embedding model
text_embedding, image_embedding = model(text_features, image_features)
# Co-trained:1 ends here

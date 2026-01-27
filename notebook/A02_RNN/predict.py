import torch
from config import Config
from data_preprocess import DataProcessor
from models import Decoder, Encoder, Seq2Seq

config = Config()


class Predictor:
    def __init__(self, model_path, data_processor):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_processor = data_processor

        # Initialize model components with the same parameters as training
        self.encoder = Encoder(
            vocab_size=self.data_processor.eng_vacab_length,
            embed_size=config.embed_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
        )

        self.decoder = Decoder(
            vocab_size=self.data_processor.fra_vacab_length,
            embed_size=config.embed_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
        )

        self.model = Seq2Seq(self.encoder, self.decoder)

        # Load the trained model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

    def preprocess_input(self, text):
        """Preprocess input text to tensor"""
        text = self.data_processor._normalize_text(text)
        words = text.split(" ")
        words.append("<EOS>")

        # Convert words to indices
        idxs = []
        for word in words:
            idx = self.data_processor.eng_word2idx.get(
                word, self.data_processor.eng_word2idx["<UNK>"]
            )
            idxs.append(idx)

        tensor = torch.tensor(idxs, device=self.device, dtype=torch.long).unsqueeze(
            0
        )  # Add batch dimension
        return tensor

    def postprocess_output(self, token_ids):
        """Convert token IDs to French sentence"""
        words = []
        for idx in token_ids:
            word = self.data_processor.fra_idx2word[idx]
            if word == "<EOS>":
                break
            if word not in ["<BOS>", "<PAD>"]:
                words.append(word)

        return " ".join(words)

    def predict(self, english_input):
        """Predict French translation for English input"""
        with torch.no_grad():
            input_tensor = self.preprocess_input(english_input)
            _, hidden = self.encoder(input_tensor)
            decoder_input = torch.tensor(
                [[self.data_processor.fra_word2idx["<BOS>"]]],
                device=self.device,
                dtype=torch.long,
            )

            output_tokens = []
            max_length = config.max_len  # Maximum length of output sequence
            for _ in range(max_length):
                decoder_output, hidden = self.decoder(decoder_input, hidden)

                # Get the next word prediction
                predicted_token = torch.argmax(decoder_output, dim=-1).unsqueeze(1)

                # Add to output sequence
                token_id = predicted_token.squeeze().item()
                output_tokens.append(token_id)

                # Check if we've reached the end of sequence
                if token_id == self.data_processor.fra_word2idx["<EOS>"]:
                    break

                # Use the predicted token as the next input
                decoder_input = predicted_token

            # Convert token IDs to French sentence
            return self.postprocess_output(output_tokens)


def main():
    # Initialize data processor and load vocabulary
    data_processor = DataProcessor(config.file_path)
    data_processor.run()  # Load and build vocabularies

    # Create predictor instance
    predictor = Predictor("./best_model.pth", data_processor)

    # Example usage
    english_input = input("Enter English text to translate to French: ")
    french_translation = predictor.predict(english_input)
    print(f"French translation: {french_translation}")


if __name__ == "__main__":
    main()

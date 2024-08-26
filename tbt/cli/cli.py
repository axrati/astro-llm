from tbt.trainer.trainer import Trainer
from tbt.model.model import DataTransformerModel
import torch.nn.functional as F
import torch
import sys

class CLI:
    def __init__(self, model:DataTransformerModel,trainer:Trainer):
        self.model = model
        self.trainer = trainer
    
    def run(self,user_input):
        cont = True
        while cont:
            if user_input=="exit":
                cont=False
                sys.exit()
            elif user_input=="help":
                self.help()
            elif user_input=="train":
                loops = int(input("# How many loops would you like to do? (#): "))
                self.trainer.train(loops)
                print("\n")
                print("Please enter a new command.")
            elif user_input=="generate":
                count = int(input("# Number of records: "))
                temp = float(input("# Temperature: "))
                self.generate(count, temp)
            else:
                a=0
                # print(user_input)
            new_input = input("# ")
            self.run(new_input)
        return None
    
    def start(self):
        """
        Starts the CLI session
        """
        print("Welcome to the trainer CLI. Type help for more information.")
        user_input = input("# ")
        self.run(user_input)        

    def generate(self, number, temperature):
        # Get the last index source/target from Trainer data and use that
        source = self.trainer.source[len(self.trainer.source)-1]
        target = self.trainer.target[len(self.trainer.target)-1]
        output = self.generate_sequence(self.model,target,source, number)
        view = input("# Would you like these printed? (y/n): ")
        if view=="y":
            for i in output:
                print(i)
        return output
    def generate_sequence(self, model, initial_target, source, N, temperature=1.0):
        current_output = source
        previous_output = initial_target
        generated_sequence = []
        self.model.eval()
        for _ in range(N):
            # Step 1: Generate output using the current target
            # if isinstance(target,dict):
            #     target = [target]
            output = model([current_output], [previous_output]) 
            # Step 2: Decode the output to get predictions
            predictions = model.decode_output(output)
            original_prediction = predictions['original']
            # Save the predictions to the generated sequence list
            generated_sequence.append(original_prediction)
            # Step 3: Update the target with the new predictions
            # Here, we need to format the predictions to be used as the next target
            current_output = previous_output
            previous_output = {key: original_prediction[key] for key in original_prediction.keys()}
        return generated_sequence


    def help(self):
        print("""
╔═══════════════════════════════════════════════════════════════════╗
║                          Trainer CLI Help                         ║
╠═══════════════════════════════════════════════════════════════════╣
║  Commands:                                                        ║
║                                                                   ║
║  ➤  `train`    : Guide you through training implementation        ║
║                  using the loaded data.                           ║
║                                                                   ║
║  ➤  `generate` : Import saved models from the `/checkpoints`      ║
║                  directory, relative to where this command is run.║
║                                                                   ║
║  ➤  `load`     : Load a JSON dataset. The dataset must be         ║
║                  labeled as { "source": [], "target": [] }.       ║
║                                                                   ║
║  ➤  `export`   : Export the trained models to a `/checkpoints`    ║
║                  directory, relative to where this command is run.║
║                                                                   ║
║  ➤  `load-`    : Import saved models from the `/checkpoints`      ║
║                  directory, relative to where this command is run.║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
    """)
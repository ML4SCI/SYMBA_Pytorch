clear

torchrun symba_trainer.py --experiment_name="_" --optimizer_type="adam" --optimizer_lr=0.0003 --optimizer_weight_decay=0.0001 \
                        --batch_size=2 --epoch=5 --dataset_name="QED_Amplitude" --model_name="bart-base" --vocab_size=52000 --num_encoder_layers=6 \
                        --num_decoder_layers=6 --embedding_size=768 --dropout=0.1 --hidden_dim=3072 --num_head=12 --maximum_sequence_length=3072 \
                        --tokenizer_type="bart-base" --distributed=True --debug=True 

clear
export CUDA_VISIBLE_DEVICES=0

python symba_trainer.py --experiment_name="_" --optimizer_lr=0.0001 --batch_size=16 \
                        --epoch=30 --dataset_name="QCD_Amplitude" --model_name="seq2seq_transformer" \
                        --vocab_size=2875 --embedding_size=512 --hidden_dim=16384 --num_head=8 \
                        --num_encoder_layers=3 --num_decoder_layers=3 --dropout=0.5 

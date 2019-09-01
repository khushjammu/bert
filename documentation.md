## SQuAD 
### Experimentation Phase 1

Finally doing BERT for TPUs. This is going to be hard. Let's get right into it. 

#### Trial 1

```bash
python run_squad.py \
  --vocab_file=gs://khush_ee/bert/wwm_cased_L-24_H-1024_A-16/vocab.txt \
  --bert_config_file=gs://khush_ee/bert/wwm_cased_L-24_H-1024_A-16/bert_config.json \
  --init_checkpoint=gs://khush_ee/bert/wwm_cased_L-24_H-1024_A-16/bert_model.ckpt \
  --do_train=True \
  --train_file=/home/jammu55048/xlnet/data/squad/train-v2.0.json \
  --do_predict=True \
  --predict_file=/home/jammu55048/xlnet/data/squad/dev-v2.0.json \
  --train_batch_size=24 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=gs://khush_ee/bert/experimentation_phase_1/trial_1/squad_large/ \
  --use_tpu=True \
  --tpu_name=$TPU_NAME \
  --version_2_with_negative=True4
```

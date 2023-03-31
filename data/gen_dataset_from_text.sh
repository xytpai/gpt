input_file=$1

python gen_dataset_from_text.py \
    --input=${input_file} \
    --len=${DATA_TEXT_LEN}

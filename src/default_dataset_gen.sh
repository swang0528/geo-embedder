# Generate the training dataset at a custom path
echo "generating training dataset..."
python dataset_gen.py --num-pairs 50000 --out-file "C:\Users\Siyang_Wang_work\Documents\A_IndependentResearch\GenAI\LayoutML\Geo-Embedder\dataset\run_dataset_062025\cgem_train.pkl"

# Generate the validation dataset at a custom path
echo "generating validation dataset..."
python dataset_gen.py --num-pairs 5000 --out-file "C:\Users\Siyang_Wang_work\Documents\A_IndependentResearch\GenAI\LayoutML\Geo-Embedder\dataset\run_dataset_062025\cgem_val.pkl"
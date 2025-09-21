
    # prepare metrics
    bleu = evaluate.load("bleu")
    #cider = evaluate.load("cider")
    rouge = evaluate.load("rouge") #rouge_raw
    meteor = evaluate.load("meteor")
    #spice = evaluate.load("spice")
    bertscore = evaluate.load("bertscore")
    #clipscore = evaluate.load("clip_score")

    # Simple metrics: we log eval loss; (optional) can add CIDEr later.
    def compute_metrics(_eval_pred):
        processor = AutoProcessor.from_pretrained("your-model-checkpoint")
        predictions, labels = _eval_pred
        # Decode predictions and labels
        decoded_preds = processor.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)
        
        # Prepare references for metrics (e.g., CIDEr and SPICE may require list of lists)
        references = [[label] for label in decoded_labels]  # For metrics expecting list of references per prediction
        
        # Compute metrics
        bleu_results = bleu.compute(predictions=decoded_preds, references=references)
        meteor_results = meteor.compute(predictions=decoded_preds, references=decoded_labels)
        rouge_results = rouge.compute(predictions=decoded_preds, references=decoded_labels)
        #cider_results = cider.compute(predictions=decoded_preds, references=references)
        #spice_results = spice.compute(predictions=decoded_preds, references=references)
        bertscore_results = bertscore.compute(predictions=decoded_preds, references=decoded_labels)
        #clipscore_results = clipscore.compute(predictions=decoded_preds, references=decoded_labels, model_type="ViT-L-14/openai")

        return {
            "bleu": bleu_results["bleu"],
            "meteor": meteor_results["meteor"],
            "rougeL": rouge_results["rougeL"],
            #"cider": cider_results["cider"],
            #"spice": spice_results["spice"],
            "bertscore": bertscore_results["bertscore"],
            #"clipscore": clipscore_results["clipscore"],
        }
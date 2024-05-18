# Set dataset:
local dataset = "musique";
local retrieval_corpus_name = dataset;
local add_pinned_paras = if dataset == "iirc" then true else false;
local valid_qids = ["2hop__804754_52230", "2hop__292995_8796", "2hop__496817_701819", "2hop__154225_727337", "2hop__642271_608104", "2hop__439265_539716", "2hop__195347_20661", "2hop__131516_53573", "2hop__427213_79175", "3hop1__443556_763924_573834", "2hop__782642_52667", "2hop__861128_15822", "4hop3__703974_789671_24078_24137", "3hop1__61746_67065_43617", "4hop3__463724_100414_35260_54090"];
local prompt_reader_args = {
    "filter_by_key_values": {
        "qid": valid_qids
    },
    "order_by_key": "qid",
    "estimated_generation_length": 300,
    "shuffle": false,
    "model_length_limit": 8000,
};

# (Potentially) Hyper-parameters:
# null means it's unused.
local llm_retrieval_count = null;
local llm_map_count = null;
local bm25_retrieval_count = 2;
local rc_context_type_ = "gold_with_n_distractors"; # Choices: no, gold, gold_with_n_distractors
local distractor_count = "1"; # Choices: 1, 2, 3
local rc_context_type = (
    if rc_context_type_ == "gold_with_n_distractors"
    then "gold_with_" + distractor_count + "_distractors"  else rc_context_type_
);
local multi_step_show_titles = null;
local multi_step_show_paras = null;
local multi_step_show_cot = null;
local rc_qa_type = null; # Choices: direct, cot

{
    "start_state": "step_by_step_bm25_retriever",
    "end_state": "[EOQ]",
    "models": {
        "step_by_step_bm25_retriever": {
            "name": "retrieve_and_reset_paragraphs",
            "next_model": "step_by_step_cot_reasoning_gen",
            "retrieval_type": "bm25",
            "retrieval_count": bm25_retrieval_count,
            "global_max_num_paras": 15,
            "query_source": "question_or_last_generated_sentence",
            "source_corpus_name": retrieval_corpus_name,
            "document_type": "title_paragraph_text",
            "return_pids": false,
            "cumulate_titles": true,
            "end_state": "[EOQ]",
        },
        "step_by_step_cot_reasoning_gen": {
            "name": "step_by_step_cot_gen",
            "next_model": "step_by_step_exit_controller",
            "prompt_file": "prompts/"+dataset+"/"+rc_context_type+"_context_cot_qa_codex.txt",
            "prompt_reader_args": prompt_reader_args,
            "generation_type": "sentences",
            "reset_queries_as_sentences": false,
            "add_context": true,
            "shuffle_paras": false,
            "terminal_return_type": null,
            "disable_exit": true,
            "end_state": "[EOQ]",
            "gen_model": "gpt3",
            "engine": "code-davinci-002",
            "retry_after_n_seconds": 50,
        },
        "step_by_step_exit_controller": {
            "name": "step_by_step_exit_controller",
            "next_model": "step_by_step_bm25_retriever",
            "answer_extractor_regex": ".* answer is:? (.*)\\.?",
            "answer_extractor_remove_last_fullstop": true,
            "terminal_state_next_model": null,
            "terminal_return_type": "pids",
            "global_max_num_paras": 15,
            "end_state": "[EOQ]",
        },
    },
    "reader": {
        "name": "multi_para_rc",
        "add_paras": false,
        "add_gold_paras": false,
        "add_pinned_paras": add_pinned_paras,
    },
    "prediction_type": "pids",
}
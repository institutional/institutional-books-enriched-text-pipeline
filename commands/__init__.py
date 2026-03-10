"""
Commands package for IBLLM

This includes the main commands:

    prepare_shards: convert HuggingFace Dataset to local shards
    setup_pipeline: make local models for rest of pipeline
    process_shard: run main pipeline (steps 1-11) on a shard
    retry_incomplete: retry failed books from earlier shards
    deduplicate: run deduplication on all shards
    postprocess_shard: apply postprocessing to a shard
    cleanup

This also includes individual step commands for fine-grained testing and application.
"""

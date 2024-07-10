

Copy
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test accuracy of TinyCLIP on dataset')
    # ... (your existing argument parsing code) ...

    args = parser.parse_args()
    
    main(args.imagenet_path, args.train_val, args.model_name, args.save_dir, args.hook_point_name)

    # After main processing, calculate and print percentiles
    save_path = create_save_path(args.save_dir, args.model_name, args.train_val, args.hook_point_name)
    percentile_results = calculate_percentiles(save_path)

    # Print results
    for layer, values in percentile_results.items():
        print(f"\nPercentiles for {layer}:")
        for percentile, value in values.items():
            print(f"{percentile}th percentile: {value}")
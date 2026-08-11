import hydra

def coordinate_skills():
    return

@hydra.main(version_base=None, config_path="config", config_name="ctme_coordination_cleanup_mini")
def main(config):
    print("Config:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    print("Starting skill coordination phase.")
    

if __name__ == "__main__":
    main()

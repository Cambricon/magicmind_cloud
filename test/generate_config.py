import os
import yaml

commit_message = os.getenv("CI_COMMIT_TITLE", "")
template_file = "test/data/template.yaml"
search_dirs = ["buildin/cv/classification", "buildin/cv/detection", "buildin/cv/segmentation", "buildin/cv/other", "buildin/nlp/LanguageModeling", "buildin/nlp/SpeechSynthesis", "buildin/nlp/SpeechRecognition", "buildin/nlp/Recommendation"]
dirs = []
for search_dir in search_dirs:
    s_dir = os.listdir(search_dir)
    for d in s_dir:
        dir = os.path.join(search_dir, d)
        # MagicMind r1.7 and later versions do not support TensorFlow Parser.
        if os.path.isdir(dir) and not d.endswith('_tensorflow'):
            dirs.append(dir)

with open(template_file, "r") as f:
    cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

for dir in dirs:
    project_name = os.path.split(dir)[-1]

    #Check if CI_COMMIT_TITLE contains "-!project_name"
    if "-!" + project_name in commit_message:
        #If included, skip the current network
        continue

# If "-" not in commit message or "-skip_ci" in commit message, ci will not be executed
    cfg[f"{project_name}"] = {
        "extends": ".network_test",
        "variables": {
           "TEST_PROJ_DIR": f"{dir}"
        },
        "rules": [
            {"if": f"$CI_COMMIT_TITLE =~ /-{project_name}/"},
            {"if": "$CI_COMMIT_TITLE =~ /-ci_test/"}
        ]
    }

with open("jobs.yml", "w", encoding="utf-8") as f:
    yaml.dump(data=cfg, stream=f, allow_unicode=True)

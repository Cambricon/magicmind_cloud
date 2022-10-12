import os
import yaml

template_file = "test/data/template.yaml"
search_dirs = ["buildin/cv/classification", "buildin/cv/detection", "buildin/cv/segmentation", "buildin/cv/other", "buildin/nlp/LanguageModeling", "buildin/nlp/SpeechSynthesis"]

dirs = []
for search_dir in search_dirs:
    s_dir = os.listdir(search_dir)
    for d in s_dir:
        dir = os.path.join(search_dir, d)
        if os.path.isdir(dir):
            dirs.append(dir)

with open(template_file, "r") as f:
    cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

for dir in dirs:
    project_name = os.path.split(dir)[-1]
    cfg[f"{project_name}"] = {
        "extends": ".network_test",
        "variables": {
           "TEST_PROJ_DIR": f"{dir}"
        }, 
        "rules": [{"if": "$CI_COMMIT_TITLE =~ /-"f"{project_name}""$/"}, {"if": "$CI_COMMIT_TITLE =~ /-ci_test$/"}]
    }

with open("jobs.yml", "w", encoding="utf-8") as f:
    yaml.dump(data=cfg, stream=f, allow_unicode=True)

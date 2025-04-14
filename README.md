
#  LLM Bias Summary Detection


Ce projet vise à détecter et atténuer les biais générés par les modèles de langage (LLM) dans les résumés de réunions, d’échanges ou de documents professionnels.  
Il combine génération, évaluation et stratégies de mitigation telles que le prompt engineering, le roleplay et le fine-tuning.

##  Objectifs

- Proposer un protocole d’évaluation des biais dans les résumés générés par des LLM.
- Implémenter des métriques d’évaluation : factualité, neutralité, polarité, diversité lexicale.
- Tester plusieurs stratégies de mitigation : Prompt Engineering, Roleplay, Fine-tuning.
- Fournir une application Streamlit (en cours de développement) pour visualiser les biais en temps réel.


## 📁 Structure du projet

```
llm-bias-summary-detection/
├── src/
│   ├── benchmark/              # Évaluation automatique (ROUGE, BLEU, polarité, etc.)
│   │   ├── eval_metrics.py
│   │   └── protocols.py
│   ├── inference/              # Génération de résumés
│   │   └── summarize.py
│   ├── mitigation/             # Stratégies de mitigation
│   │   ├── fine_tuning.py
│   │   ├── prompt_engineering.py
│   │   └── roleplay_strategies.py
├── data/                       # Données d’entraînement et de test
│   └── fine_tune_dataset.csv
├── models/                     # Modèles fine-tunés sauvegardés
├── tests/                      # Tests unitaires avec pytest
│   ├── test_inference.py
│  
├── run_fine_tune.py            # Script CLI d’entraînement
├── run_tests.sh                # Script d’exécution des tests
├── requirements.txt
└── README.md
```


##  Usage

###  Fine-tuner un modèle de résumé (avec ou sans LoRA)

```bash
python run_fine_tune.py \
  --data_path data/fine_tune_dataset.csv \
  --model_name facebook/bart-large-cnn \
  --output_dir models/fine_tuned_bart \
  --epochs 3 \
  --batch_size 4 \
  --use_peft
```

- `--data_path` : chemin vers un fichier `.csv` contenant les colonnes `text` et `summary`  
- `--model_name` : nom du modèle Hugging Face à fine-tuner  
- `--output_dir` : dossier de sauvegarde du modèle entraîné  
- `--use_peft` : (optionnel) active un fine-tuning léger avec LoRA




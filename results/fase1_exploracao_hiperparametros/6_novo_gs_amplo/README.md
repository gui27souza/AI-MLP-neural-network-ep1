# Novo Grid Search Amplo

- Grid search buscando analisar o máximo de combinações possíveis entre hiperparâmetros (variando nos extremos do espectro muito baixo, normal, muito alto)

- LSS - senti na pele o problema de não ter implementado um sistema de salvamento robusto ao longo da execução do grid-search. Fiz uma cirurgia em runtime e consegui recuperar o csv que só seria gerado no final, mas precisei abrir mão das análises com o hiperparâmetro de VAL_RATIO em 0.45

- Comando que me salvou:

```
sudo gdb -p 5789 -batch \
-ex 'call (void) PyRun_SimpleString("import gc; [r.save_csv() for r in gc.get_objects() if type(r).__name__ == \"GridSearchReporter\"]")' \
-ex 'detach' \
-ex 'quit'
```

> Basciamente pausa o código por um período milimetricamente curto, salva o csv que está <br>
> apenas em memória acionando o método que só seria acionado ao fim do grid-search completo

- No quesitos de resultados sobre o melhor modelo, sem novidades
- No quesito de estudo de relacionamento entre os hiperparâmetros, ficou sensacional

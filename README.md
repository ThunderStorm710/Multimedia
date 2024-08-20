# Multimédia - Trabalhos Práticos

## Trabalho Prático nº 1: Compressão de Imagem

### Objectivo
O objetivo deste trabalho é desenvolver uma compreensão fundamental sobre a compressão de imagem, especialmente utilizando o codec JPEG.

### Planeamento
- **Prazo de Entrega:** 17 de Março, sexta-feira, 23h59
- **Esforço Extra-Aulas:** 18 horas por aluno
- **Formato de Entrega:** Arquivo ZIP contendo o código completo e o relatório em formato PDF. Para evitar erros na submissão, adicionar a extensão `.pdf` ao nome do arquivo ZIP (e.g., `arquivo.zip.pdf`).

### Estrutura do Trabalho

1. **Compressão de Imagens BMP no Formato JPEG**
   - Compressão de imagens com qualidade alta, média e baixa usando um editor de imagem (e.g., GIMP, Adobe Photoshop).
   - Comparar os resultados obtidos.

2. **Funções Encoder e Decoder**
   - Criar funções para encapsular as funcionalidades desenvolvidas nos pontos seguintes.

3. **Visualização de Imagens no Modelo de Cor RGB**
   - Leitura e manipulação de imagens BMP, separação dos canais RGB e visualização dos mesmos.

4. **Pré-processamento da Imagem: Padding**
   - Implementação de padding para garantir que a imagem tenha dimensões múltiplas de 32x32 pixels.

5. **Conversão para o Modelo de Cor YCbCr**
   - Converter a imagem do modelo de cor RGB para YCbCr e vice-versa. Comparar os canais Y com R, G, B e com Cb e Cr.

6. **Sub-amostragem**
   - Sub-amostragem dos canais Y, Cb e Cr conforme definido pelo codec JPEG e análise da taxa de compressão.

7. **Transformada de Coseno Discreta (DCT)**
   - Aplicar a DCT aos canais Y, Cb, Cr tanto nos canais completos quanto em blocos de 8x8 e 64x64 pixels. Comparar os resultados em termos de potencial de compressão.

8. **Quantização**
   - Quantizar os coeficientes da DCT com diferentes fatores de qualidade e discutir os resultados.

9. **Codificação DPCM dos Coeficientes DC**
   - Implementar a codificação diferencial dos coeficientes DC e analisar os resultados.

10. **Codificação e Descodificação End-to-End**
    - Implementar a codificação e descodificação completas, visualizar as imagens descodificadas e calcular métricas de distorção como MSE, RMSE, SNR e PSNR.

---

## Trabalho Prático nº 2: Music Information Retrieval

### Objectivo
O objetivo deste trabalho é adquirir sensibilidade para questões fundamentais de Multimedia Information Retrieval, especialmente em sistemas de recomendação de música baseados em conteúdo.

### Planeamento
- **Prazo de Entrega:** 12 de Maio, sexta-feira, 23h59
- **Esforço Extra-Aulas:** 18 horas por aluno
- **Formato de Entrega:** Arquivo ZIP contendo o código completo e o relatório em formato PDF. Adicionar a extensão `.pdf` ao nome do arquivo ZIP (e.g., `arquivo.zip.pdf`).

### Estrutura do Trabalho

1. **Preparação**
   - Familiarizar-se com sistemas de recomendação de música (e.g., Jango.com, Spotify, Last.fm).
   - Descarregar e analisar a base de dados 4Q audio emotion dataset.
   - Estudar e utilizar a framework de processamento áudio `librosa`.

2. **Extracção de Features**
   - Processar as features do dataset e normalizá-las.
   - Extrair features utilizando a framework `librosa` e calcular estatísticas como média, desvio padrão, skewness, curtose, entre outras.

3. **Implementação de Métricas de Similaridade**
   - Desenvolver código para calcular distâncias Euclidiana, de Manhattan, e do Coseno.
   - Gerar matrizes de similaridade e rankings de similaridade para as queries fornecidas.

4. **Avaliação**
   - **Avaliação Objectiva:** Analisar a correspondência das recomendações com metadados como artista, género, quadrante e emoção. Calcular a métrica `precision`.
   - **Avaliação Subjectiva:** Avaliar a qualidade das recomendações utilizando uma escala de Likert, calcular médias, desvio padrão e precision.

5. **Alínea com Bonificação (Opcional)**
   - Implementar features de raiz sem usar `librosa` ou outras bibliotecas de alto nível.

---

## Entrega do Projeto

Os projetos devem ser entregues na plataforma InforEstudante até as datas limites especificadas. O arquivo ZIP deve conter:
- O código completo desenvolvido.
- Relatório detalhado em formato PDF.
- Todos os arquivos necessários para replicar os resultados.

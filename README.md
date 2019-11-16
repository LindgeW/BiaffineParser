# BiaffineParser
Biaffine Parser based on CNN / LSTM / Transformer Encoder

> Results on CTB 5.1
<table align='center'>
  <tr>
    <th></th>
    <th colspan='2'>Baseline(without punctuation)</th>
    <th colspan='2'>my performance(without punctuation)</th>
    <th colspan='2'>my performance(with punctuation)</th>
  </tr>
  <tr>
    <td>UAS</td>
    <td>LAS</td>
  </tr>
  <tr>
    <td>89.30</td>
    <td>88.23</td>
  </tr>
  <tr>
    <td><b>89.774</b></td>
    <td><b>88.357</b></td>
  </tr>
  <tr>
    <td>88.551</td>
    <td>87.323</td>
  </tr>
</table>
For evaluation, omitting punctuations will get better performance in uas and las.
We can also prove this conclusion from mathematics.

# BiaffineParser
Biaffine Parser based on CNN / LSTM / Transformer Encoder

> Results on CTB 5.1
<table>
  <tr>
    <th></th>
    <th>Baseline<br>(without punctuation)</th>
    <th>my performance<br>(without punctuation)</th>
    <th>my performance<br>(with punctuation)</th>
    <th>*Bert(Finetune-last)</th>
    <th>*Bert(Fixed-8)</th>
  </tr>
  <tr>
    <td>UAS</td>
    <td>89.30</td>
    <td><b>89.774</b></td>
    <td>88.551</td>
    <td>92.52</td>
    <td>92.96</td>
  </tr>
  <tr>
    <td>LAS</td>
    <td>88.23</td>
    <td><b>88.357</b></td>
    <td>87.323</td>
    <td>91.23</td>
    <td>91.80</td>
  </tr>
</table>
For evaluation, omitting punctuations will get better performance in uas and las.
We can also prove this conclusion from mathematics.

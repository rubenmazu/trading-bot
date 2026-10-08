# Requirements Document

## Introduction

`Quant_Trading_System` este un sistem real de cercetare și trading algoritmic bazat exclusiv pe reguli matematice sau statistice deterministe, versionate și explicabile. Sistemul urmărește identificarea unor avantaje mici și repetabile, evaluate net de toate costurile relevante, fără garanții de profit sau de evitare a pierderilor. Orizontul inițial este de 5–60 de minute și exclude HFT.

Prima etapă folosește un capital Live de referință de 100 EUR, o pierdere totală maximă acceptată de 10 EUR și un risc țintă de 0,25–0,50 EUR per tranzacție numai când dimensiunea minimă și costurile instrumentului permit. Futures, CFD, levierul și comenzile reale sunt excluse din etapa inițială. Același motor production-grade trebuie utilizat în Backtest, Shadow, Demo și, numai după îndeplinirea tuturor barierelor, Live.

Brokerul, universul inițial exact, sursa de date, stack-ul tehnic și durata minimă Demo rămân decizii deschise. Cerințele stabilesc criterii verificabile pentru rezolvarea ulterioară a acestor decizii fără a reduce siguranța. Documentul nu reprezintă consultanță financiară și nu promite performanță viitoare.

## Glossary

- **Quant_Trading_System**: Ansamblul software definit de acest document.
- **Trading_Engine**: Motorul comun care procesează evenimente, semnale, risc, ordine și execuții în toate mediile.
- **Operational_Mode**: Unul dintre modurile Backtest, Shadow, Demo sau Live.
- **Initial_Stage**: Etapa proiectului în care transmiterea ordinelor reale este interzisă.
- **Backtest**: Redare event-driven a datelor istorice, fără conexiune de tranzacționare reală.
- **Shadow**: Procesare a datelor curente cu execuții simulate local.
- **Demo**: Procesare conectată exclusiv la contul demonstrativ al brokerului.
- **Live**: Procesare care poate transmite ordine reale numai după ieșirea aprobată din Initial_Stage și satisfacerea Live_Gate.
- **Broker_Adapter**: Interfață care izolează Trading_Engine de API-ul și modelele unui broker.
- **Data_Adapter**: Interfață care izolează Market_Data_Subsystem de o sursă de date.
- **Market_Data_Subsystem**: Componentă care ingerează, validează, normalizează și distribuie date de piață.
- **Market_Event**: Observație de piață normalizată, cu timpul sursei și timpul recepției.
- **Bar**: Agregare OHLCV pentru un interval temporal configurat.
- **Instrument_Universe**: Set versionat de instrumente eligibile.
- **Strategy**: Set versionat de reguli matematice sau statistice deterministe.
- **Signal**: Rezultatul explicabil produs de Strategy.
- **Strategy_Artifact**: Pachet imuabil cu Strategy, parametri, date, configurații, cod și rezultate de validare.
- **Order_Intent**: Intenție internă de ordin înaintea aprobării de risc.
- **Real_Order_Request**: Cerere care ar putea produce o tranzacție într-un cont real.
- **Order_Management_Subsystem**: Componentă care gestionează ciclul de viață al ordinelor.
- **Execution_Event**: Confirmare, respingere, anulare, expirare sau execuție parțială ori totală.
- **Risk_Engine**: Componentă independentă care aprobă, reduce sau respinge Order_Intent.
- **Reconciliation_Subsystem**: Componentă care compară starea internă cu starea brokerului.
- **Kill_Switch**: Control care blochează ordinele noi și aplică politica ordinelor deschise.
- **Complete_Cost_Model**: Model versionat pentru spread, comisioane, slippage, latență, conversie valutară și taxe aplicabile configurate.
- **Trading_Day**: Zi de tranzacționare conform calendarului și fusului orar al pieței.
- **Out_Of_Sample_Set**: Date excluse din calibrare și selecție.
- **Walk_Forward_Validation**: Evaluare cu ferestre de calibrare urmate temporal de ferestre Out_Of_Sample_Set.
- **Bootstrap_Monte_Carlo_Analysis**: Reeșantionare și simulare a distribuției rezultatelor nete.
- **Market_Regime**: Categorie de piață definită prin praguri măsurabile și versionate.
- **Multiple_Testing_Correction**: Metodă statistică preînregistrată pentru ajustarea evaluării ipotezelor multiple.
- **Promotion_Criteria**: Praguri numerice versionate și aprobate înaintea evaluării unei etape.
- **Paper_Qualification**: Calificare cumulată în Shadow și Demo înainte de eligibilitatea Live.
- **Live_Gate**: Ansamblul indivizibil de condiții care trebuie satisfăcute înaintea oricărei transmisii Live.
- **Configuration_Snapshot**: Copie imuabilă a configurației efective a unei rulări.
- **Audit_Record**: Înregistrare corelată și protejată contra modificării nedetectate.
- **Idempotency_Key**: Identificator persistent care previne aplicarea duplicată a unei operații.
- **Recovery_Point**: Stare persistentă verificată de la care procesarea poate continua.
- **Secret_Store**: Mecanism autorizat pentru stocarea și accesarea credentialelor fără includerea valorilor secrete în cod sau configurații obișnuite.
- **Health_State**: Stare calculată din prospețimea datelor, conectivitate, reconciliere, ceas și dependențe.
- **Supported_Load**: Volumul maxim versionat de instrumente și evenimente pentru care sunt validate țintele de performanță.
- **Open_Decision**: Decizie nerezolvată, cu opțiuni, criterii, responsabil și stare documentate.
- **HFT**: Trading de înaltă frecvență dependent de reacții de ordinul milisecundelor.
- **OHLCV**: Valorile open, high, low, close și volume ale unui Bar.
- **API**: Interfață programatică oferită de un serviciu extern.
- **Endpoint**: Adresă configurată a unui API pentru un mediu determinat.
- **ETF**: Fond tranzacționat la bursă.
- **FX**: Piața valutară.
- **Crypto**: Clasă separată de active digitale lichide.
- **Futures**: Contracte standardizate cu decontare viitoare.
- **CFD**: Contracte pentru diferență fără deținerea activului suport.
- **Drawdown**: Scăderea valorii portofoliului de la un maxim anterior la minimul ulterior.
- **Development_Set**: Date permise pentru proiectarea, calibrarea și selectarea Strategy.
- **Critical_Incident**: Incident care afectează ordinele, pozițiile, soldurile, riscul, securitatea sau integritatea auditului.
- **Fail_Safe_Block**: Barieră independentă de respingerea primară care împiedică Real_Order_Request să ajungă la Broker_Adapter.
- **Capital_Change_Authorization**: Predicat unic adevărat numai când aprobarea explicită este validă și noul Configuration_Snapshot este creat atomic.

## Requirements

### Requirement 1: Motor comun și separarea mediilor

**User Story:** Ca operator, vreau același motor în toate modurile, pentru ca validarea și operarea să fie comparabile.

#### Acceptance Criteria

1. THE Trading_Engine SHALL aplica aceeași succesiune de reguli pentru Market_Event, Signal, Order_Intent și Execution_Event în fiecare Operational_Mode.
2. WHEN un Operational_Mode este selectat, THE Trading_Engine SHALL utiliza exclusiv adaptoarele și credentialele asociate modului selectat.
3. IF configurația unui Operational_Mode este incompletă, THEN THE Quant_Trading_System SHALL refuza pornirea și raporta fiecare câmp lipsă.
4. THE Quant_Trading_System SHALL înregistra versiunea Trading_Engine pentru fiecare rulare.
5. WHEN Operational_Mode este schimbat, THE Quant_Trading_System SHALL păstra nemodificate codul Strategy, parametrii și regulile Risk_Engine, schimbând numai adaptoarele, configurația de mediu și referințele credentialelor.

### Requirement 2: Interdicția comenzilor reale în etapa inițială

**User Story:** Ca proprietar al capitalului, vreau blocarea tehnică a ordinelor reale în etapa inițială, pentru a preveni tranzacționarea accidentală.

#### Acceptance Criteria

1. WHILE etapa proiectului este Initial_Stage, THE Quant_Trading_System SHALL respinge fiecare Real_Order_Request înainte de Broker_Adapter.
2. WHILE etapa proiectului este Initial_Stage, THE Quant_Trading_System SHALL permite numai Backtest, Shadow și Demo.
3. WHEN Quant_Trading_System este instalat, actualizat, restaurat sau migrat, THE Quant_Trading_System SHALL seta Live ca dezactivat.
4. IF o configurație Initial_Stage conține endpoint sau cont Live, THEN THE Quant_Trading_System SHALL refuza pornirea.
5. WHEN o încercare Real_Order_Request este respinsă în Initial_Stage, THE Quant_Trading_System SHALL crea un Audit_Record cu motivul respingerii.
6. IF respingerea unei Real_Order_Request eșuează în Initial_Stage, THEN THE Fail_Safe_Block SHALL împiedica sincron cererea să ajungă la Broker_Adapter și crea un Audit_Record pentru tentativa eșuată.

### Requirement 3: Independența și selecția brokerului

**User Story:** Ca operator, vreau brokeri interschimbabili și o selecție bazată pe dovezi, pentru a evita dependența de un furnizor nepotrivit.

#### Acceptance Criteria

1. THE Broker_Adapter SHALL transforma mesajele brokerului în modele normalizate Market_Event și Execution_Event.
2. THE Broker_Adapter SHALL transforma ordinele aprobate în cereri compatibile cu brokerul configurat.
3. WHEN Broker_Adapter este înlocuit, THE Strategy SHALL păstra regulile și parametrii Strategy_Artifact.
4. IF brokerul nu oferă o capabilitate solicitată, THEN THE Broker_Adapter SHALL respinge cererea cu un cod de motiv.
5. WHEN brokerul este evaluat, THE Quant_Trading_System SHALL înregistra accesul pentru rezidenți din România, disponibilitatea API, mediul Demo, costurile, fracțiunile, activele eligibile și fiabilitatea documentată.
6. WHILE Open_Decision pentru broker este nerezolvată, THE Quant_Trading_System SHALL bloca eligibilitatea Live.

### Requirement 4: Univers eligibil și orizont operațional

**User Story:** Ca cercetător, vreau un univers lichid, fără levier și cu intervale realiste, pentru a limita riscul și complexitatea inițială.

#### Acceptance Criteria

1. THE Instrument_Universe SHALL permite candidați din indici accesibili fără derivate, aur sau mărfuri accesibile fără derivate, acțiuni sau ETF-uri lichide și FX major accesibil fără levier.
2. WHERE evaluarea crypto este activată, THE Instrument_Universe SHALL izola crypto într-o configurație, un buget de risc și un raport distincte.
3. THE Instrument_Universe SHALL exclude futures, CFD, opțiuni complexe, altcoins și instrumente ilichide în Initial_Stage.
4. THE Instrument_Universe SHALL exclude orice expunere care necesită levier sau vânzare în lipsă în Initial_Stage.
5. THE Strategy SHALL utiliza intervale Bar configurate între 5 și 60 de minute inclusiv în Initial_Stage.
6. WHEN un instrument este evaluat, THE Quant_Trading_System SHALL aplica praguri versionate pentru volum, spread, valoare minimă, fracționare, monedă și program de tranzacționare.
7. IF un instrument nu permite respectarea bugetului per tranzacție după costuri, THEN THE Quant_Trading_System SHALL declara instrumentul neeligibil.

### Requirement 5: Ingestia și integritatea datelor

**User Story:** Ca cercetător, vreau date istorice și curente validate și trasabile, pentru ca rezultatele să nu depindă de intrări corupte.

#### Acceptance Criteria

1. THE Market_Data_Subsystem SHALL ingera date istorice și date curente prin Data_Adapter.
2. WHEN un Market_Event este ingerat, THE Market_Data_Subsystem SHALL păstra sursa, instrumentul, timpul sursei, timpul recepției și numărul de secvență disponibil.
3. THE Market_Data_Subsystem SHALL utiliza aceeași schemă normalizată în toate Operational_Mode.
4. IF un Bar are câmpuri obligatorii absente, valori OHLCV imposibile sau ordine temporală invalidă, THEN THE Market_Data_Subsystem SHALL exclude Bar din Strategy și înregistra motivul.
5. IF prospețimea ultimului Market_Event depășește pragul configurat pentru instrument, THEN THE Market_Data_Subsystem SHALL bloca Order_Intent pentru instrument.
6. WHEN un set de date este salvat, THE Market_Data_Subsystem SHALL înregistra identificatorul sursei, intervalul temporal, fusul orar, calendarul, ajustările corporative și suma de control.
7. WHEN același eveniment este ingerat repetat, THE Market_Data_Subsystem SHALL păstra o singură reprezentare canonică.
8. IF verificarea sumei de control eșuează, THEN THE Market_Data_Subsystem SHALL invalida setul de date afectat.

### Requirement 6: Strategii deterministe și explicabile

**User Story:** Ca cercetător, vreau reguli deterministe și explicabile, pentru a reproduce și justifica fiecare semnal.

#### Acceptance Criteria

1. WHEN Strategy primește aceleași intrări, aceeași stare inițială și aceiași parametri, THE Strategy SHALL produce aceeași secvență Signal.
2. THE Strategy SHALL exprima intrările, ieșirile și dimensionarea prin reguli matematice sau statistice versionate.
3. WHEN Strategy produce Signal, THE Strategy SHALL furniza intrările, regulile evaluate, parametrii și codul motivului.
4. IF o intrare necesară este absentă sau invalidă, THEN THE Strategy SHALL produce un Signal fără acțiune și un cod de motiv.
5. THE Quant_Trading_System SHALL asocia fiecare Signal cu versiunea Strategy, Configuration_Snapshot și identificatorii datelor.
6. THE Strategy SHALL exclude modele care produc decizii imposibil de reconstruit din reguli și intrări înregistrate.

### Requirement 7: Backtest event-driven și fără look-ahead

**User Story:** Ca cercetător, vreau un Backtest event-driven, pentru ca simularea să respecte ordinea informației disponibile.

#### Acceptance Criteria

1. WHILE Operational_Mode este Backtest, THE Trading_Engine SHALL procesa Market_Event în ordine temporală conform unei priorități versionate pentru egalități.
2. WHILE Operational_Mode este Backtest, THE Strategy SHALL accesa numai date disponibile până la timpul Market_Event curent.
3. WHEN un ordin simulat devine eligibil, THE Trading_Engine SHALL genera Execution_Event conform modelului versionat de execuție.
4. IF un gol de date depășește pragul configurat, THEN THE Quant_Trading_System SHALL invalida exclusiv subperioada Backtest afectată.
5. WHEN sursa oferă istoricul componenței, THE Instrument_Universe SHALL utiliza componența și instrumentele delistate valabile la timpul simulat.
6. WHEN un Backtest este repetat cu același Strategy_Artifact, THE Quant_Trading_System SHALL produce aceleași semnale, ordine, execuții și metrici.
7. IF un gol de date este mai mic sau egal cu pragul configurat, THEN THE Quant_Trading_System SHALL continua Backtest fără acțiune asupra subperioadei.

### Requirement 8: Costuri și latență

**User Story:** Ca cercetător, vreau evaluarea completă a costurilor și latenței, pentru a măsura avantajul net în condiții realiste.

#### Acceptance Criteria

1. WHERE taxele aplicabile sunt configurate și aprobate, THE Complete_Cost_Model SHALL calcula separat spread, comisioane, slippage, efectul latenței, conversia valutară și taxele aplicabile.
2. WHEN un Execution_Event este evaluat, THE Complete_Cost_Model SHALL aplica regulile valabile pentru instrument, broker, piață, dimensiune și timp.
3. IF o componentă obligatorie a costului nu este configurată, THEN THE Quant_Trading_System SHALL invalida evaluarea Strategy.
4. THE Quant_Trading_System SHALL raporta separat rezultatul brut, fiecare categorie de cost și rezultatul net.
5. IF avantajul estimat este mai mic sau egal cu zero după Complete_Cost_Model, THEN THE Quant_Trading_System SHALL respinge Strategy.
6. THE Quant_Trading_System SHALL exclude din Initial_Stage ipotezele de execuție HFT sau cu reacție sub o secundă ca sursă obligatorie a avantajului.

### Requirement 9: Ciclul de viață al ordinelor

**User Story:** Ca operator, vreau stări explicite pentru ordine, pentru ca portofoliul intern să reflecte execuțiile confirmate.

#### Acceptance Criteria

1. THE Order_Management_Subsystem SHALL defini stări pentru creare, aprobare, transmitere, confirmare, respingere, execuție parțială, execuție totală, anulare, expirare și rezultat necunoscut.
2. WHEN un Execution_Event este primit, THE Order_Management_Subsystem SHALL valida tranziția față de starea curentă.
3. WHEN o execuție parțială este confirmată, THE Order_Management_Subsystem SHALL actualiza cantitatea executată, cantitatea rămasă, prețul mediu și costurile acumulate.
4. IF o tranziție este invalidă, THEN THE Order_Management_Subsystem SHALL respinge exclusiv tranziția și păstra fără recalculare starea și cantitatea confirmate anterior.
5. WHEN o anulare este confirmată sau respinsă, THE Order_Management_Subsystem SHALL înregistra rezultatul și cantitatea rămasă executabilă.
6. WHILE un ordin are un incident de stare nerezolvat, THE Order_Management_Subsystem SHALL îngheța toate modificările ordinului.

### Requirement 10: Idempotență și protecție contra duplicatelor

**User Story:** Ca operator, vreau procesare idempotentă, pentru ca retransmisiile să nu dubleze ordinele sau execuțiile.

#### Acceptance Criteria

1. WHEN un Order_Intent este aprobat, THE Order_Management_Subsystem SHALL atribui un Idempotency_Key unic și persistent.
2. WHEN același Idempotency_Key este procesat repetat, THE Order_Management_Subsystem SHALL aplica efectul operației o singură dată.
3. WHEN un Execution_Event duplicat este primit, THE Order_Management_Subsystem SHALL păstra neschimbate poziția, numerarul și costurile după prima aplicare.
4. IF secvența mesajelor are un gol detectabil, THEN THE Order_Management_Subsystem SHALL suspenda exclusiv procesarea dependentă de gol, continua mesajele independente și solicita mesajele lipsă.
5. WHEN o retransmisie este detectată, THE Order_Management_Subsystem SHALL corela retransmisia cu operația inițială în Audit_Record.

### Requirement 11: Reconectare și reconciliere

**User Story:** Ca operator, vreau recuperarea stării brokerului înaintea ordinelor noi, pentru a evita decizii bazate pe poziții incorecte.

#### Acceptance Criteria

1. WHEN conexiunea brokerului este restabilită, THE Broker_Adapter SHALL recupera ordinele, execuțiile, pozițiile și soldurile înaintea unui ordin nou.
2. WHEN Quant_Trading_System pornește în Demo sau Live, THE Reconciliation_Subsystem SHALL compara starea internă cu starea brokerului.
3. WHILE Operational_Mode este Demo sau Live, THE Reconciliation_Subsystem SHALL rula la un interval configurat de maximum 60 de secunde.
4. WHEN brokerul raportează Execution_Event, THE Reconciliation_Subsystem SHALL iniția imediat reconcilierea ordinului și instrumentului asociate indiferent de starea recuperării.
5. IF o diferență depășește toleranța versionată, THEN THE Reconciliation_Subsystem SHALL bloca sincron cu detectarea ordinele noi exclusiv pentru instrumentul individual afectat și crea un Audit_Record.
6. IF starea brokerului nu poate fi recuperată complet, THEN THE Quant_Trading_System SHALL activa Kill_Switch.
7. WHEN reluarea după o diferență este solicitată, THE Reconciliation_Subsystem SHALL cere împreună cauza, corecția și aprobarea operatorului.

### Requirement 12: Motor de risc independent

**User Story:** Ca proprietar al capitalului, vreau risc independent de Strategy, pentru ca semnalele să nu poată ocoli limitele.

#### Acceptance Criteria

1. THE Risk_Engine SHALL evalua fiecare Order_Intent după Signal și înainte de Broker_Adapter.
2. THE Strategy SHALL exclude autoritatea de aprobare sau modificare a limitelor Risk_Engine.
3. WHEN un Order_Intent este evaluat, THE Risk_Engine SHALL calcula pierderea la nivelul de protecție, expunerea, numerarul, costurile și expunerea agregată.
4. IF o valoare calculată depășește limita activă, THEN THE Risk_Engine SHALL respinge Order_Intent și înregistra valoarea calculată și limita.
5. THE Risk_Engine SHALL aplica aceleași reguli versionate în toate Operational_Mode.
6. IF datele necesare calculului sunt absente, invalide sau expirate, THEN THE Risk_Engine SHALL respinge Order_Intent.
7. WHEN Risk_Engine reduce dimensiunea Order_Intent, THE Risk_Engine SHALL reevalua toate limitele pentru dimensiunea redusă.

### Requirement 13: Bugete de risc și limitarea pierderilor

**User Story:** Ca proprietar al capitalului, vreau limite măsurabile pe tranzacție, zi și total, pentru a limita pierderea acceptată.

#### Acceptance Criteria

1. WHILE capitalul Live de referință este 100 EUR, THE Risk_Engine SHALL configura riscul țintă per tranzacție între 0,25 EUR și 0,50 EUR inclusiv.
2. IF pierderea estimată a Order_Intent depășește 0,50 EUR, THEN THE Risk_Engine SHALL respinge Order_Intent.
3. THE Risk_Engine SHALL configura înaintea fiecărui Trading_Day o limită zilnică mai mică sau egală cu 2 EUR.
4. IF pierderea netă realizată plus pierderea nerealizată atinge limita zilnică, THEN THE Risk_Engine SHALL activa Kill_Switch pentru Trading_Day curent.
5. THE Risk_Engine SHALL utiliza 10 EUR drept limită totală de pierdere pentru capitalul Live de referință.
6. IF capitalul curent plus retragerile cumulate minus depunerile ulterioare este mai mic sau egal cu 90 EUR, THEN THE Risk_Engine SHALL activa Kill_Switch permanent pentru configurația de capital curentă.
7. IF Order_Intent necesită levier, împrumut de numerar, futures, CFD sau vânzare în lipsă, THEN THE Risk_Engine SHALL respinge Order_Intent în Initial_Stage.
8. IF dimensiunea minimă depășește riscul permis sau numerarul disponibil după costuri, THEN THE Risk_Engine SHALL respinge Order_Intent.
9. THE Risk_Engine SHALL include costurile estimate în toate limitele monetare.
10. WHERE etapa proiectului este ulterioară Initial_Stage, THE Risk_Engine SHALL permite evaluarea Order_Intent anterior interzise numai conform unei configurații de risc aprobate.
11. WHILE limita totală de pierdere este atinsă, THE Quant_Trading_System SHALL permite reluarea numai printr-o configurație de capital nouă care satisface Capital_Change_Authorization conform cerinței 28, fără ca aprobarea operatorului să poată dezactiva Kill_Switch în configurația existentă.

### Requirement 14: Kill switch

**User Story:** Ca operator, vreau oprire automată și manuală, pentru a limita activitatea în condiții nesigure.

#### Acceptance Criteria

1. WHEN operatorul activează Kill_Switch, THE Quant_Trading_System SHALL bloca ordinele noi în maximum o secundă.
2. WHEN o regulă automată activează Kill_Switch, THE Quant_Trading_System SHALL bloca ordinele noi în maximum o secundă.
3. WHEN Kill_Switch este activat, THE Order_Management_Subsystem SHALL aplica politica configurată de păstrare sau anulare fiecărui ordin deschis.
4. WHILE Kill_Switch este activ, THE Quant_Trading_System SHALL continua recepția Execution_Event, reconcilierea și auditul.
5. WHEN reluarea este solicitată, THE Quant_Trading_System SHALL cere reconciliere reușită și aprobare explicită a operatorului.
6. IF motivul activării rămâne nerezolvat, THEN THE Quant_Trading_System SHALL păstra Kill_Switch activ.
7. WHEN politica ordinelor deschise nu este configurată, THE Order_Management_Subsystem SHALL păstra implicit ordinele deschise fără modificare.
8. IF reconcilierea nu reușește, THEN THE Quant_Trading_System SHALL refuza reluarea indiferent de aprobarea operatorului.

### Requirement 15: Bariere explicite pentru Live

**User Story:** Ca operator, vreau un Live_Gate indivizibil, pentru ca nicio condiție de siguranță să nu fie omisă.

#### Acceptance Criteria

1. WHEN activarea Live este solicitată, THE Live_Gate SHALL cere ieșirea aprobată din Initial_Stage.
2. WHEN activarea Live este solicitată, THE Live_Gate SHALL cere Open_Decision pentru broker în stare aprobată.
3. WHEN activarea Live este solicitată, THE Live_Gate SHALL cere un Strategy_Artifact eligibil și o Paper_Qualification validă.
4. WHEN activarea Live este solicitată, THE Live_Gate SHALL cere reconciliere reușită, Health_State sănătoasă și zero Critical_Incident nerezolvate.
5. WHEN activarea Live este solicitată, THE Live_Gate SHALL cere limite Risk_Engine aprobate și credentiale asociate contului Live aprobat.
6. WHEN toate condițiile Live_Gate sunt satisfăcute, THE Quant_Trading_System SHALL cere două confirmări explicite distincte ale operatorului.
7. WHEN prima confirmare Live este înregistrată, THE Quant_Trading_System SHALL accepta a doua confirmare la maximum 10 minute inclusiv și invalida activarea după depășirea intervalului.
8. IF o condiție Live_Gate este falsă, THEN THE Quant_Trading_System SHALL respinge activarea Live cu lista condițiilor nesatisfăcute.
9. IF identitatea contului sau mediul brokerului diferă de configurația aprobată, THEN THE Broker_Adapter SHALL respinge Real_Order_Request.
10. WHEN Quant_Trading_System repornește, THE Quant_Trading_System SHALL suspenda Live până la reconciliere și reconfirmarea operatorului.
11. WHEN activarea Live este prezentată, THE Quant_Trading_System SHALL afișa avertizarea că pierderile sunt posibile și profitul nu este garantat.
12. WHEN începe o nouă tentativă de confirmare Live, THE Live_Gate SHALL reevalua de la zero fiecare condiție de activare.
13. IF o confirmare Live expiră sau este invalidată, THEN THE Quant_Trading_System SHALL invalida toate confirmările tentativei și cere reluarea întregului flux Live_Gate.

### Requirement 16: Promovare controlată între medii

**User Story:** Ca operator, vreau promovarea aceluiași artefact, pentru ca logica validată să fie logica operată.

#### Acceptance Criteria

1. THE Quant_Trading_System SHALL permite promovarea numai în ordinea Backtest, Shadow, Demo și Live.
2. WHEN Strategy_Artifact este promovat, THE Quant_Trading_System SHALL păstra Strategy, parametrii, regulile Risk_Engine și Complete_Cost_Model.
3. IF Strategy_Artifact se modifică după aprobare, THEN THE Quant_Trading_System SHALL crea o versiune nouă și relua validarea de la Backtest.
4. WHEN o etapă este finalizată, THE Quant_Trading_System SHALL înregistra criteriile, rezultatul, identitatea aprobatorului și timpul aprobării.
5. IF etapa nu îndeplinește Promotion_Criteria preînregistrate, THEN THE Quant_Trading_System SHALL bloca etapa următoare.
6. WHEN un artefact este promovat, THE Quant_Trading_System SHALL schimba numai adaptoarele, configurația de mediu și referințele credentialelor.
7. IF promovarea detectează diferențe în afara adaptoarelor, configurației de mediu sau referințelor credentialelor, THEN THE Quant_Trading_System SHALL bloca promovarea până la eliminarea diferențelor.
8. WHEN promovarea este blocată sau respinsă, THE Quant_Trading_System SHALL înregistra starea terminală Respins.

### Requirement 17: Configurare și reproductibilitate

**User Story:** Ca cercetător, vreau configurații validate și rulări reproductibile, pentru a reconstrui fiecare rezultat.

#### Acceptance Criteria

1. THE Quant_Trading_System SHALL valida fiecare configurație față de o schemă versionată înaintea pornirii.
2. IF o configurație conține câmpuri necunoscute sau valori în afara limitelor, THEN THE Quant_Trading_System SHALL refuza pornirea și raporta fiecare abatere.
3. WHEN o rulare începe, THE Quant_Trading_System SHALL crea un Configuration_Snapshot cu versiunea codului, Strategy_Artifact, datele, fusul orar și configurația efectivă.
4. WHEN aleatorietatea este utilizată în analiză, THE Quant_Trading_System SHALL înregistra algoritmul și sămânța pseudoaleatoare.
5. WHEN o analiză este repetată cu același Configuration_Snapshot și aceleași date, THE Quant_Trading_System SHALL produce rezultate identice în limitele numerice versionate.
6. THE Quant_Trading_System SHALL separa configurațiile Backtest, Shadow, Demo și Live prin identificatori expliciți de mediu.
7. IF Configuration_Snapshot nu poate fi creat, THEN THE Quant_Trading_System SHALL opri rularea înaintea procesării.

### Requirement 18: Separare development/test și out-of-sample

**User Story:** Ca cercetător, vreau separare temporală între calibrare și evaluare, pentru a limita contaminarea rezultatelor.

#### Acceptance Criteria

1. THE Quant_Trading_System SHALL separa cronologic Development_Set de Out_Of_Sample_Set.
2. WHILE Strategy sau parametrii sunt selectați, THE Quant_Trading_System SHALL limita accesul selecției la Development_Set.
3. WHEN Strategy și parametrii sunt fixați, THE Quant_Trading_System SHALL evalua o singură dată Out_Of_Sample_Set rezervat deciziei curente.
4. IF Out_Of_Sample_Set influențează modificarea Strategy, THEN THE Quant_Trading_System SHALL reclasifica datele consultate ca Development_Set și rezerva un nou Out_Of_Sample_Set ulterior.
5. THE Strategy_Artifact SHALL păstra limitele temporale, identificatorii și rolul fiecărei partiții.
6. IF un Out_Of_Sample_Set dedicat nu este rezervat și evaluat exact o dată, THEN THE Quant_Trading_System SHALL bloca finalizarea evaluării.

### Requirement 19: Walk-forward, bootstrap și Monte Carlo

**User Story:** Ca cercetător, vreau evaluări repetate și distribuții ale riscului, pentru a detecta rezultate dependente de o singură perioadă.

#### Acceptance Criteria

1. THE Walk_Forward_Validation SHALL evalua minimum cinci ferestre Out_Of_Sample_Set ulterioare ferestrelor Development_Set asociate.
2. WHEN ferestrele sunt definite, THE Quant_Trading_System SHALL fixa lungimea, pasul și recalibrarea înaintea evaluării finale.
3. THE Bootstrap_Monte_Carlo_Analysis SHALL executa minimum 10.000 de reeșantionări ale rezultatelor nete.
4. WHEN Bootstrap_Monte_Carlo_Analysis se încheie, THE Quant_Trading_System SHALL raporta distribuțiile rezultatului net, drawdown-ului maxim, seriilor de pierderi și atingerii limitei totale.
5. WHEN robustețea parametrilor este evaluată, THE Quant_Trading_System SHALL compara parametrul ales cu vecinătățile definite înaintea evaluării.
6. IF rezultatul acceptabil apare numai într-un punct izolat al spațiului parametrilor, THEN THE Quant_Trading_System SHALL respinge Strategy.
7. WHILE Bootstrap_Monte_Carlo_Analysis este în desfășurare, THE Quant_Trading_System SHALL permite raportarea parțială a fiecărei distribuții disponibile.
8. IF lungimea, pasul sau regula de recalibrare Walk_Forward_Validation nu este fixată, THEN THE Quant_Trading_System SHALL bloca evaluarea.

### Requirement 20: Regimuri și stresarea ipotezelor

**User Story:** Ca cercetător, vreau evaluare pe regimuri și costuri degradate, pentru a măsura fragilitatea avantajului.

#### Acceptance Criteria

1. THE Quant_Trading_System SHALL defini Market_Regime prin praguri măsurabile înaintea evaluării finale.
2. WHEN Strategy este evaluată, THE Quant_Trading_System SHALL raporta rezultatele nete separat pentru fiecare Market_Regime prezent.
3. THE Quant_Trading_System SHALL evalua spread, comisioane și slippage la nivelul de bază și la multiplicatori de 1,5 și 2,0.
4. THE Quant_Trading_System SHALL evalua latența la nivelul de bază și la valorile de stres preînregistrate.
5. IF datele de validare conțin zero sau un Market_Regime, THEN THE Quant_Trading_System SHALL bloca promovarea.
6. IF Strategy nu îndeplinește Promotion_Criteria la stresul preînregistrat, THEN THE Quant_Trading_System SHALL respinge Strategy.

### Requirement 21: Controlul supraoptimizării și testării multiple

**User Story:** Ca cercetător, vreau preînregistrarea ipotezelor și corecție statistică, pentru a reduce selecția rezultatelor întâmplătoare.

#### Acceptance Criteria

1. THE Quant_Trading_System SHALL înregistra Promotion_Criteria, metrica principală, spațiul parametrilor și numărul variantelor înaintea evaluării finale.
2. WHEN sunt comparate mai multe variante, THE Multiple_Testing_Correction SHALL aplica metoda versionată și preînregistrată.
3. THE Quant_Trading_System SHALL raporta fiecare variantă evaluată, inclusiv variantele respinse.
4. IF rezultatul net cumulat Out_Of_Sample_Set este mai mic sau egal cu zero, THEN THE Quant_Trading_System SHALL respinge Strategy.
5. IF Strategy nu îndeplinește pragul ajustat prin Multiple_Testing_Correction, THEN THE Quant_Trading_System SHALL respinge Strategy.
6. IF eliminarea numărului preînregistrat de tranzacții cu cele mai mari câștiguri face rezultatul net mai mic sau egal cu zero, THEN THE Quant_Trading_System SHALL respinge Strategy.
7. WHEN Strategy este acceptată, THE Quant_Trading_System SHALL raporta incertitudinea estimării și absența unei garanții de profit.
8. IF metoda Multiple_Testing_Correction nu este preînregistrată, THEN THE Quant_Trading_System SHALL bloca evaluarea variantelor.

### Requirement 22: Calificare Shadow și Demo

**User Story:** Ca operator, vreau o calificare suficientă pe date curente înainte de Live, pentru a detecta diferențe operaționale.

#### Acceptance Criteria

1. WHEN Paper_Qualification este pregătită pentru începere, THE Quant_Trading_System SHALL cere Promotion_Criteria numerice pentru durata Shadow, durata Demo, numărul de ordine și acoperirea Market_Regime.
2. WHILE Open_Decision pentru durata Demo este nerezolvată, THE Quant_Trading_System SHALL bloca finalizarea Paper_Qualification.
3. WHILE Paper_Qualification este activă, THE Trading_Engine SHALL utiliza Strategy_Artifact fără modificarea Strategy sau parametrilor.
4. WHEN Paper_Qualification este evaluată, THE Quant_Trading_System SHALL compara rezultatul net, execuțiile, respingerile, deconectările și reconcilierile cu Promotion_Criteria.
5. IF rezultatul net cumulat în Shadow sau Demo este mai mic sau egal cu zero după Complete_Cost_Model, THEN THE Quant_Trading_System SHALL respinge promovarea Live.
6. IF există un Critical_Incident nerezolvat, THEN THE Quant_Trading_System SHALL bloca promovarea Live.
7. IF Strategy sau parametrii se modifică, THEN THE Quant_Trading_System SHALL invalida Paper_Qualification și relua promovarea de la Backtest.
8. WHEN durata Demo este aprobată, THE Quant_Trading_System SHALL înregistra valoarea, justificarea și aprobatorul în Open_Decision.

### Requirement 23: Securitatea credentialelor și accesului

**User Story:** Ca operator, vreau credentiale protejate și acces minim, pentru a reduce riscul utilizării neautorizate a contului.

#### Acceptance Criteria

1. THE Quant_Trading_System SHALL încărca credentialele brokerului și sursei de date numai din Secret_Store.
2. THE Quant_Trading_System SHALL exclude valorile credentialelor din cod, Configuration_Snapshot, loguri, erori și Audit_Record.
3. THE Quant_Trading_System SHALL utiliza credentiale distincte pentru Demo și Live.
4. WHEN o componentă solicită un secret, THE Secret_Store SHALL acorda acces numai identității și mediului autorizate.
5. IF Secret_Store este indisponibil sau accesul este refuzat, THEN THE Quant_Trading_System SHALL bloca numai conexiunile noi și păstra sesiunile active până la următoarea reîmprospătare a secretului.
6. WHEN credentialele sunt rotite, THE Quant_Trading_System SHALL invalida referința veche conform politicii configurate și crea un Audit_Record fără valoarea secretă.
7. IF o valoare secretă este detectată într-o ieșire persistentă, THEN THE Quant_Trading_System SHALL bloca publicarea ieșirii și activa un incident de securitate.
8. THE Quant_Trading_System SHALL cere autentificare consolidată pentru activarea Live și modificarea limitelor de risc.

### Requirement 24: Audit și trasabilitate

**User Story:** Ca operator și auditor, vreau o urmă completă și verificabilă, pentru a reconstrui fiecare decizie.

#### Acceptance Criteria

1. THE Quant_Trading_System SHALL crea Audit_Record pentru Market_Event utilizate, Signal, Order_Intent, decizii Risk_Engine, cereri broker, Execution_Event și reconciliere.
2. THE Quant_Trading_System SHALL crea Audit_Record pentru autentificări, modificări de configurație, promovări, Kill_Switch și tentative Live.
3. WHEN un Audit_Record este creat, THE Quant_Trading_System SHALL include timpul, tipul, identificatorul de corelare, versiunea componentei, actorul și rezultatul.
4. THE Quant_Trading_System SHALL proteja ordinea Audit_Record printr-un mecanism care detectează modificarea sau ștergerea.
5. WHEN o decizie este reconstruită, THE Quant_Trading_System SHALL furniza lanțul Market_Event–Signal–Order_Intent–risc–Execution_Event.
6. THE Quant_Trading_System SHALL aplica o perioadă de retenție configurată și aprobată conform obligațiilor aplicabile operatorului din România.
7. WHEN Audit_Record sunt exportate, THE Quant_Trading_System SHALL include versiunea schemei și dovada verificabilă a integrității.

### Requirement 25: Observabilitate și alertare

**User Story:** Ca operator, vreau stări și alerte măsurabile, pentru a detecta condițiile nesigure înaintea unor ordine noi.

#### Acceptance Criteria

1. THE Quant_Trading_System SHALL publica Health_State pentru date, broker, reconciliere, Risk_Engine, stocare, ceas și audit.
2. WHEN Health_State se schimbă, THE Quant_Trading_System SHALL înregistra componenta, starea nouă, cauza și timpul.
3. IF o componentă critică intră în stare nesănătoasă în Demo sau Live, THEN THE Quant_Trading_System SHALL bloca ordinele noi în maximum o secundă.
4. WHEN Kill_Switch, o depășire de risc, o deconectare sau o diferență de reconciliere apare, THE Quant_Trading_System SHALL emite o alertă în maximum cinci secunde.
5. THE Quant_Trading_System SHALL expune metrici pentru latențe, prospețimea datelor, ordine pe stare, execuții, costuri, profit și pierdere netă și utilizarea limitelor de risc.
6. IF livrarea unei alerte eșuează, THEN THE Quant_Trading_System SHALL păstra alerta pentru reîncercare și crea un Audit_Record.

### Requirement 26: Recuperare și reziliență

**User Story:** Ca operator, vreau recuperare deterministă după întreruperi, pentru ca starea financiară și limitele să rămână corecte.

#### Acceptance Criteria

1. THE Quant_Trading_System SHALL persista ordinele, execuțiile, pozițiile, soldurile, limitele Risk_Engine și starea Kill_Switch într-un Recovery_Point.
2. WHEN Quant_Trading_System repornește, THE Quant_Trading_System SHALL valida integritatea Recovery_Point înainte de Market_Event noi.
3. WHILE recuperarea este incompletă, WHEN un Recovery_Point valid este încărcat, THE Quant_Trading_System SHALL relua de la primul eveniment neconfirmat.
4. IF Recovery_Point este invalid sau incomplet, THEN THE Quant_Trading_System SHALL activa Kill_Switch și bloca ordinele în maximum o secundă.
5. WHEN reconcilierea stării brokerului reușește și fiecare ordin în așteptare are o stare confirmată de broker, THE Quant_Trading_System SHALL marca recuperarea finalizată.
6. WHILE Operational_Mode este Demo sau Live, WHEN diferența ceasului față de sursa temporală depășește 500 de milisecunde, THE Quant_Trading_System SHALL suspenda ordinele noi independent de starea recuperării.
7. WHEN o dependență indisponibilă revine, THE Quant_Trading_System SHALL utiliza reconectare cu intervale limitate și configurate înaintea reluării.
8. IF după recuperare rămân evenimente neconfirmate sau ordine cu rezultat necunoscut, THEN THE Quant_Trading_System SHALL menține blocate ordinele noi pentru instrumentele afectate până la reconcilierea lor.
9. WHEN recuperarea este marcată finalizată și Kill_Switch este inactiv, THE Quant_Trading_System SHALL debloca procesarea evenimentelor și ordinele noi.

### Requirement 27: Performanță realistă

**User Story:** Ca operator, vreau performanță măsurată pentru orizontul de minute, pentru ca întârzierile să fie controlate fără obiective HFT.

#### Acceptance Criteria

1. THE Quant_Trading_System SHALL defini Supported_Load prin număr de instrumente, frecvență de evenimente și resurse de calcul.
2. WHILE încărcarea este în Supported_Load, THE Trading_Engine SHALL finaliza procesarea Strategy și Risk_Engine pentru 95% dintre Bar în maximum o secundă de la recepție.
3. WHILE încărcarea este în Supported_Load, THE Trading_Engine SHALL finaliza procesarea Strategy și Risk_Engine pentru fiecare Bar înainte de expirarea pragului de prospețime al instrumentului.
4. IF coada Market_Event depășește pragul versionat, THEN THE Quant_Trading_System SHALL bloca ordinele noi exclusiv pentru instrumentele afectate direct de evenimentele acumulate.
5. WHEN un test de performanță este executat, THE Quant_Trading_System SHALL raporta percentilele 50, 95 și 99 pentru ingestie, Strategy, risc și gestionarea ordinelor.
6. THE Quant_Trading_System SHALL exclude ca obiectiv funcționarea HFT sau garantarea latenței de ordinul milisecundelor.

### Requirement 28: Capital și modificarea limitelor

**User Story:** Ca proprietar al capitalului, vreau creșteri controlate ale capitalului, pentru ca fondurile suplimentare să depindă de dovezi aprobate.

#### Acceptance Criteria

1. THE Quant_Trading_System SHALL configura 100 EUR drept capital Live inițial de referință.
2. THE Quant_Trading_System SHALL configura 10 EUR drept pierdere totală maximă acceptată.
3. WHEN este propus capital suplimentar, THE Quant_Trading_System SHALL cere Paper_Qualification validă și Promotion_Criteria aprobate pentru rezultate convingătoare.
4. WHEN capitalul configurat se modifică, THE Quant_Trading_System SHALL accepta modificarea numai dacă Capital_Change_Authorization este adevărat.
5. IF o modificare de capital nu include limite Risk_Engine reevaluate, THEN THE Quant_Trading_System SHALL respinge modificarea.
6. THE Quant_Trading_System SHALL exclude alimentarea automată a contului și creșterea automată a limitelor.
7. IF limita totală propusă depășește 10 EUR, THEN THE Quant_Trading_System SHALL respinge configurația în domeniul acestei specificații.

### Requirement 29: Raportarea evaluării și a performanței

**User Story:** Ca cercetător, vreau rapoarte complete și comparabile, pentru a decide pe baza avantajului net și a riscului.

#### Acceptance Criteria

1. WHEN o evaluare se încheie, THE Quant_Trading_System SHALL raporta perioada, universul, numărul tranzacțiilor, rezultatul brut, rezultatul net, costurile și drawdown-ul maxim.
2. WHEN o evaluare se încheie, THE Quant_Trading_System SHALL raporta ipotezele de latență, lichiditate, spread, slippage, comisioane, conversie și taxe configurate.
3. WHEN rezultatele sunt comparate între Operational_Mode, THE Quant_Trading_System SHALL utiliza aceeași monedă de raportare și aceleași definiții ale metricilor.
4. IF o metrică obligatorie nu poate fi calculată, THEN THE Quant_Trading_System SHALL marca raportul incomplet și bloca promovarea.
5. THE Quant_Trading_System SHALL prezenta rezultatele fără afirmații de profit garantat sau extrapolări neînsoțite de ipoteze.

### Requirement 30: Gestionarea deciziilor deschise

**User Story:** Ca responsabil de proiect, vreau decizii deschise explicite, pentru a continua specificarea fără presupuneri ascunse.

#### Acceptance Criteria

1. THE Quant_Trading_System SHALL menține Open_Decision pentru broker, universul inițial exact, sursa de date, stack-ul tehnic și durata minimă Demo.
2. WHEN o Open_Decision este creată, THE Quant_Trading_System SHALL înregistra opțiunile, criteriile măsurabile, responsabilul, termenul și starea.
3. WHEN Open_Decision pentru broker este evaluată, THE Quant_Trading_System SHALL păstra IBKR și Alpaca numai ca opțiuni preliminare până la verificarea criteriilor.
4. WHEN o Open_Decision este aprobată, THE Quant_Trading_System SHALL înregistra alegerea, dovezile, consecințele și aprobatorul.
5. IF o Open_Decision afectează cel puțin unul dintre domeniile siguranță, costuri, reproductibilitate sau eligibilitate Live, THEN THE Quant_Trading_System SHALL bloca etapa dependentă până la aprobare.
6. IF Open_Decision pentru sursa de date este nerezolvată, THEN THE Quant_Trading_System SHALL bloca validarea finală a Strategy.
7. IF Open_Decision pentru stack-ul tehnic este nerezolvată, THEN THE Quant_Trading_System SHALL bloca începerea implementării.
8. IF evaluarea brokerului păstrează zero opțiuni preliminare, THEN THE Quant_Trading_System SHALL invalida evaluarea.
9. WHEN o Open_Decision care blochează eligibilitatea Live este rezolvată, THE Live_Gate SHALL debloca automat condiția de eligibilitate asociată.

## Matrice de trasabilitate

| Domeniu obligatoriu | Cerințe |
|---|---|
| Motor unic Backtest → Shadow → Demo → Live | 1, 16 |
| Interdicția inițială și blocarea Live | 2, 15 |
| Independență și alegere broker | 3, 30 |
| Univers, lichiditate, fără levier, 5–60 minute | 4, 13 |
| Date și integritate | 5 |
| Determinism, explicabilitate, reproductibilitate | 6, 7, 17 |
| Costuri, latență, edge net | 8, 20, 27, 29 |
| Ordine parțiale, respinse, anulate | 9 |
| Idempotență și duplicate | 10 |
| Reconectare și reconciliere | 11, 26 |
| Risc, limite și kill switch | 12, 13, 14, 28 |
| Validare și anti-overfitting | 18, 19, 20, 21, 22 |
| Credentiale și acces | 23 |
| Audit și observabilitate | 24, 25 |
| Configurare și promovare | 16, 17 |
| Decizii deschise | 22, 30 |

## Decizii deschise

| Decizie | Stare | Opțiuni sau domeniu curent | Condiție de rezolvare |
|---|---|---|---|
| Broker | Deschisă | IBKR și Alpaca sunt candidați preliminari, fără presupunerea eligibilității | Verificarea criteriilor 3.5 pentru România și aprobarea dovezilor |
| Univers inițial exact | Deschisă | Instrumente lichide eligibile din clasele permise de cerința 4 | Date, costuri, fracțiuni și respectarea riscului pentru 100 EUR |
| Sursă de date | Deschisă | Furnizor istoric și realtime independent prin Data_Adapter | Acoperire, calitate, licență, cost și timestamp-uri validate |
| Stack tehnic | Propusă în design.md (Python 3.13, uv, pydantic, SQLite, hypothesis) | Limbaj, stocare, infrastructură și librării | Aprobarea design.md înaintea Tasks |
| Durată minimă Demo | Deschisă | Prag numeric care va completa volumul și regimurile Paper_Qualification | Justificare statistică și operațională aprobată înainte de Live |

## Note de conformitate a cerințelor

- Fiecare criteriu este identificabil prin perechea `Cerința.Criteriul` (de exemplu, 15.8) pentru trasabilitate în Design și Tasks.
- Pragurile financiare sunt limite de risc, nu obiective sau garanții de profit.
- Orice obligație fiscală sau de retenție trebuie confirmată pentru situația concretă a operatorului înainte de Live; sistemul tratează valorile aplicabile drept configurații versionate.

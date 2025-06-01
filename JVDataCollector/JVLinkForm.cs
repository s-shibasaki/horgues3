using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Npgsql;

namespace JVDataCollector
{
     public partial class JVLinkForm : Form
    {
        public class RecordDefinition
        {
            public string TableName { get; set; }
            public string RecordTypeId { get; set; }
            public List<FieldDefinition> Fields { get; set; } = new List<FieldDefinition>();
            public FieldDefinition creationDateField { get; set; }
            public string Comment { get; set; } = string.Empty;
        }

        public class FieldDefinition
        {
            public int Position { get; set; }
            public int Length { get; set; }
            public string Comment { get; set; } = string.Empty;
        }

        public class NormalFieldDefinition : FieldDefinition
        {
            public string ColumnName { get; set; }
            public bool IsPrimaryKey { get; set; } = false;
        }

        public class RepeatFieldDefinition : FieldDefinition
        {
            public string TableName { get; set; }
            public int RepeatCount { get; set; }
            public List<FieldDefinition> Fields { get; set; } = new List<FieldDefinition>();
        }

        private string[] commandLineArgs;
        private string connectionString = "Host=localhost;Database=horgues3;Username=postgres;Password=postgres";
        private List<RecordDefinition> recordDefinitions = new List<RecordDefinition>();

        public JVLinkForm(string[] args)
        {
            InitializeComponent();
            commandLineArgs = args;
        }

        private void JVLinkForm_Load(object sender, EventArgs e)
        {
            try
            {
                // レコード定義の作成
                CreateRecordDefinitions();

                Console.WriteLine($"JVDataCollector started at {DateTime.Now:yyyy-MM-dd HH:mm:ss}");

                if (commandLineArgs.Length > 0)
                {
                    string command = commandLineArgs[0].ToLower();
                    Console.WriteLine($"Executing command: {command}");

                    switch (command)
                    {
                        case "setup":
                            ExecuteSetup();
                            break;
                        case "update":
                            ExecuteUpdate();
                            break;
                        default:
                            throw new ArgumentException($"Unknown command: {command}");
                    }
                }
                else
                {
                    throw new ArgumentException("No command line arguments provided. Usage: program.exe [setup|update]");
                }

                Console.WriteLine($"JVDataCollector completed successfully at {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
                Console.WriteLine($"Stack Trace:");
                Console.WriteLine(ex.StackTrace);
                Console.WriteLine($"JVDataCollector failed at {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
            }
            finally
            {
                Application.Exit();
            }
        }

        private void ProcessJVData(string dataSpec, string fromTime, int option)
        {
            Console.WriteLine($"Processing JV data...");
            Console.WriteLine($"  DataSpec: {dataSpec}");
            Console.WriteLine($"  FromTime: {fromTime}");
            Console.WriteLine($"  Option: {option}");

            int result;
            int filesToRead = 0, filesToDownload = 0;
            string lastFileTimestamp;

            result = axJVLink1.JVOpen(dataSpec, fromTime, option, ref filesToRead, ref filesToDownload, out lastFileTimestamp);
            if (result != 0)
            {
                throw new InvalidOperationException($"JVOpen ({dataSpec}) failed with error code: {result}");
            }

            Console.WriteLine($"JVOpen executed successfully:");
            Console.WriteLine($"  FilesToRead: {filesToRead}");
            Console.WriteLine($"  FilesToDownload: {filesToDownload}");
            Console.WriteLine($"  LastFileTimestamp: {lastFileTimestamp}");

            // JVStatusでダウンロード進捗を監視
            int downloadedFiles = 0;
            int previousDownloadedFiles = -1;

            while (downloadedFiles < filesToDownload)
            {
                // 少し待ってからJVStatusを呼び出し
                System.Threading.Thread.Sleep(1000); // 1秒待機

                downloadedFiles = axJVLink1.JVStatus();
                if (downloadedFiles < 0)
                {
                    throw new InvalidOperationException($"JVStatus failed with error code: {downloadedFiles}");
                }

                // ダウンロード済みファイル数が変化したときのみコンソール出力
                if (downloadedFiles != previousDownloadedFiles)
                {
                    Console.WriteLine($"Downloaded files: {downloadedFiles}/{filesToDownload}");
                    previousDownloadedFiles = downloadedFiles;
                }
            }

            Console.WriteLine($"All files downloaded successfully.");

            // TODO: JVRead を行う

            // TODO: lastFileTimestamp をデータベースに保存する

            axJVLink1.JVClose();
            Console.WriteLine($"JV data processing completed.");
        }

        private void ExecuteSetup()
        {
            CreateTables();

            Console.WriteLine("Initializing JVLink...");
            int result = axJVLink1.JVInit("SA000000/SD000004");
            if (result != 0)
            {
                throw new InvalidOperationException($"JVInit failed with error code: {result}");
            }
            Console.WriteLine("JVLink initialized successfully.");

            // 読み出し終了ポイントを指定できないデータ種別
            StringBuilder sb = new StringBuilder();
            sb.Append("TOKU");
            sb.Append("DIFN");
            sb.Append("HOSN");
            sb.Append("HOYU");
            sb.Append("COMM");
            string dataSpec = sb.ToString();
            string fromTime = "19860101000000";
            ProcessJVData(dataSpec, fromTime, 4);

            // 読み出し終了ポイントを指定できるデータ種別は1年分ずつ取得
            sb.Clear();
            sb.Append("RACE");
            sb.Append("SLOP");
            sb.Append("WOOD");
            sb.Append("YSCH");
            sb.Append("MING");
            sb.Append("BLDN");
            sb.Append("SNPN");
            dataSpec = sb.ToString();

            // 1986年から去年まで1年ずつ取得
            int currentYear = DateTime.Now.Year;
            for (int year = 1986; year < currentYear; year++)
            {
                fromTime = $"{year}0101000000-{year + 1}0101000000";
                ProcessJVData(dataSpec, fromTime, 4);
            }

            // 今年の分は終了時刻を指定しない
            fromTime = $"{currentYear}0101000000";
            ProcessJVData(dataSpec, fromTime, 4);
            
            Console.WriteLine("Setup process completed.");
        }

        private void CreateTables()
        {
            Console.WriteLine("Creating database tables...");
            
            using (var connection = new NpgsqlConnection(connectionString))
            {
                connection.Open();
                
                foreach (var recordDef in recordDefinitions)
                {
                    CreateTableRecursive(connection, recordDef.TableName, recordDef.Fields, null);
                }
            }
            
            Console.WriteLine("Database tables created successfully.");
        }

        private void CreateTableRecursive(NpgsqlConnection connection, string tableName, List<FieldDefinition> fields, List<NormalFieldDefinition> parentPrimaryKeys)
        {
            // 既存のテーブルを削除
            var dropTableSql = $"DROP TABLE IF EXISTS {tableName}";
            using (var dropCommand = new NpgsqlCommand(dropTableSql, connection))
            {
                dropCommand.ExecuteNonQuery();
            }
            
            var columns = new List<string>();
            var primaryKeys = new List<string>();
            
            // 親テーブルの主キーを継承
            if (parentPrimaryKeys != null)
            {
                foreach (var pk in parentPrimaryKeys)
                {
                    columns.Add($"{pk.ColumnName} CHAR({pk.Length})");
                    primaryKeys.Add(pk.ColumnName);
                }
                // 繰り返し番号を追加
                columns.Add("repeat_index INTEGER");
                primaryKeys.Add("repeat_index");
            }
            
            var currentPrimaryKeys = parentPrimaryKeys?.ToList() ?? new List<NormalFieldDefinition>();
            
            foreach (var field in fields)
            {
                if (field is NormalFieldDefinition normalField)
                {
                    columns.Add($"{normalField.ColumnName} CHAR({normalField.Length})");
                    if (normalField.IsPrimaryKey)
                    {
                        primaryKeys.Add(normalField.ColumnName);
                        currentPrimaryKeys.Add(normalField);
                    }
                }
                else if (field is RepeatFieldDefinition repeatField)
                {
                    // 子テーブルを再帰的に作成
                    CreateTableRecursive(connection, repeatField.TableName, repeatField.Fields, currentPrimaryKeys);
                }
            }
            
            // データ作成日付カラムを追加
            columns.Add("created_date DATE NOT NULL");
            
            var createTableSql = $"CREATE TABLE {tableName} ({string.Join(", ", columns)}";
            if (primaryKeys.Count > 0)
            {
                createTableSql += $", PRIMARY KEY ({string.Join(", ", primaryKeys)})";
            }
            createTableSql += ")";
            
            using (var command = new NpgsqlCommand(createTableSql, connection))
            {
                command.ExecuteNonQuery();
            }
            
            Console.WriteLine($"Table '{tableName}' created.");
        }

        private void ExecuteUpdate()
        {
            throw new NotImplementedException("Update functionality is not yet implemented");
        }

        private void CreateRecordDefinitions()
        {
            // 特別登録馬レコード定義
            recordDefinitions.Add(
                new RecordDefinition
                {
                    TableName = "special_registration",
                    RecordTypeId = "TK",
                    Comment = "特別登録馬レコード - ハンデ発表前後の特別競走登録馬情報",
                    Fields = new List<FieldDefinition>
                    {
                        new NormalFieldDefinition { Position = 1, ColumnName = "record_type_id", Length = 2, IsPrimaryKey = false, Comment = "レコード種別ID - 'TK' をセット" },
                        new NormalFieldDefinition { Position = 3, ColumnName = "data_kubun", Length = 1, IsPrimaryKey = false, Comment = "データ区分 - 1:ハンデ発表前(通常日曜) 2:ハンデ発表後(通常月曜) 0:該当レコード削除" },
                        new NormalFieldDefinition { Position = 12, ColumnName = "kaisai_year", Length = 4, IsPrimaryKey = true, Comment = "開催年 - 該当レース施行年 西暦4桁 yyyy形式" },
                        new NormalFieldDefinition { Position = 16, ColumnName = "kaisai_month_day", Length = 4, IsPrimaryKey = true, Comment = "開催月日 - 該当レース施行月日 各2桁 mmdd形式" },
                        new NormalFieldDefinition { Position = 20, ColumnName = "track_code", Length = 2, IsPrimaryKey = true, Comment = "競馬場コード - 該当レース施行競馬場 <コード表 2001.競馬場コード>参照" },
                        new NormalFieldDefinition { Position = 22, ColumnName = "kaisai_kai", Length = 2, IsPrimaryKey = true, Comment = "開催回[第N回] - 該当レース施行回 その競馬場でその年の何回目の開催かを示す" },
                        new NormalFieldDefinition { Position = 24, ColumnName = "kaisai_day", Length = 2, IsPrimaryKey = true, Comment = "開催日目[N日目] - そのレース施行回で何日目の開催かを示す" },
                        new NormalFieldDefinition { Position = 26, ColumnName = "race_number", Length = 2, IsPrimaryKey = true, Comment = "レース番号 - 該当レース番号" },
                        new NormalFieldDefinition { Position = 28, ColumnName = "day_of_week_code", Length = 1, IsPrimaryKey = false, Comment = "曜日コード - 該当レース施行曜日 <コード表 2002.曜日コード>参照" },
                        new NormalFieldDefinition { Position = 29, ColumnName = "special_race_number", Length = 4, IsPrimaryKey = false, Comment = "特別競走番号 - 重賞レースのみ設定 原則的には過去の同一レースと一致する番号" },
                        new NormalFieldDefinition { Position = 33, ColumnName = "race_name_main", Length = 60, IsPrimaryKey = false, Comment = "競走名本題 - 全角30文字 レース名の本題" },
                        new NormalFieldDefinition { Position = 93, ColumnName = "race_name_sub", Length = 60, IsPrimaryKey = false, Comment = "競走名副題 - 全角30文字 レース名の副題（スポンサー名や記念名など）" },
                        new NormalFieldDefinition { Position = 153, ColumnName = "race_name_kakko", Length = 60, IsPrimaryKey = false, Comment = "競走名カッコ内 - 全角30文字 レースの条件やトライアル対象レース名、レース名通称など" },
                        new NormalFieldDefinition { Position = 213, ColumnName = "race_name_main_eng", Length = 120, IsPrimaryKey = false, Comment = "競走名本題欧字 - 半角120文字" },
                        new NormalFieldDefinition { Position = 333, ColumnName = "race_name_sub_eng", Length = 120, IsPrimaryKey = false, Comment = "競走名副題欧字 - 半角120文字" },
                        new NormalFieldDefinition { Position = 453, ColumnName = "race_name_kakko_eng", Length = 120, IsPrimaryKey = false, Comment = "競走名カッコ内欧字 - 半角120文字" },
                        new NormalFieldDefinition { Position = 573, ColumnName = "race_name_short10", Length = 20, IsPrimaryKey = false, Comment = "競走名略称10文字 - 全角10文字" },
                        new NormalFieldDefinition { Position = 593, ColumnName = "race_name_short6", Length = 12, IsPrimaryKey = false, Comment = "競走名略称6文字 - 全角6文字" },
                        new NormalFieldDefinition { Position = 605, ColumnName = "race_name_short3", Length = 6, IsPrimaryKey = false, Comment = "競走名略称3文字 - 全角3文字" },
                        new NormalFieldDefinition { Position = 611, ColumnName = "race_name_kubun", Length = 1, IsPrimaryKey = false, Comment = "競走名区分 - 重賞回次を本題・副題・カッコ内のうちどれに設定すべきかを示す (0:初期値 1:本題 2:副題 3:カッコ内)" },
                        new NormalFieldDefinition { Position = 612, ColumnName = "jyusho_kaiji", Length = 3, IsPrimaryKey = false, Comment = "重賞回次[第N回] - そのレースの重賞としての通算回数を示す" },
                        new NormalFieldDefinition { Position = 615, ColumnName = "grade_code", Length = 1, IsPrimaryKey = false, Comment = "グレードコード - <コード表 2003.グレードコード>参照" },
                        new NormalFieldDefinition { Position = 616, ColumnName = "race_type_code", Length = 2, IsPrimaryKey = false, Comment = "競走種別コード - <コード表 2005.競走種別コード>参照" },
                        new NormalFieldDefinition { Position = 618, ColumnName = "race_symbol_code", Length = 3, IsPrimaryKey = false, Comment = "競走記号コード - <コード表 2006.競走記号コード>参照" },
                        new NormalFieldDefinition { Position = 621, ColumnName = "weight_type_code", Length = 1, IsPrimaryKey = false, Comment = "重量種別コード - <コード表 2008.重量種別コード>参照" },
                        new NormalFieldDefinition { Position = 622, ColumnName = "race_condition_2yo", Length = 3, IsPrimaryKey = false, Comment = "競走条件コード 2歳条件 - <コード表 2007.競走条件コード>参照" },
                        new NormalFieldDefinition { Position = 625, ColumnName = "race_condition_3yo", Length = 3, IsPrimaryKey = false, Comment = "競走条件コード 3歳条件 - <コード表 2007.競走条件コード>参照" },
                        new NormalFieldDefinition { Position = 628, ColumnName = "race_condition_4yo", Length = 3, IsPrimaryKey = false, Comment = "競走条件コード 4歳条件 - <コード表 2007.競走条件コード>参照" },
                        new NormalFieldDefinition { Position = 631, ColumnName = "race_condition_5yo_up", Length = 3, IsPrimaryKey = false, Comment = "競走条件コード 5歳以上条件 - <コード表 2007.競走条件コード>参照" },
                        new NormalFieldDefinition { Position = 634, ColumnName = "race_condition_youngest", Length = 3, IsPrimaryKey = false, Comment = "競走条件コード 最若年条件 - 出走可能な最も馬齢が若い馬に対する条件 <コード表 2007.競走条件コード>参照" },
                        new NormalFieldDefinition { Position = 637, ColumnName = "distance", Length = 4, IsPrimaryKey = false, Comment = "距離 - 単位:メートル" },
                        new NormalFieldDefinition { Position = 641, ColumnName = "track_code_detail", Length = 2, IsPrimaryKey = false, Comment = "トラックコード - <コード表 2009.トラックコード>参照" },
                        new NormalFieldDefinition { Position = 643, ColumnName = "course_kubun", Length = 2, IsPrimaryKey = false, Comment = "コース区分 - 半角2文字 使用するコース A～E を設定" },
                        new NormalFieldDefinition { Position = 645, ColumnName = "handicap_announce_date", Length = 8, IsPrimaryKey = false, Comment = "ハンデ発表日 - ハンデキャップレースにおいてハンデが発表された日 yyyymmdd 形式" },
                        new NormalFieldDefinition { Position = 653, ColumnName = "registration_count", Length = 3, IsPrimaryKey = false, Comment = "登録頭数 - 特別登録された頭数" },
                        new RepeatFieldDefinition
                        {
                            Position = 656,
                            TableName = "special_registration_horses",
                            RepeatCount = 300,
                            Length = 70,
                            Comment = "登録馬毎情報 - 特別登録された各馬の詳細情報",
                            Fields = new List<FieldDefinition>
                            {
                                new NormalFieldDefinition { Position = 1, ColumnName = "sequence_number", Length = 3, IsPrimaryKey = false, Comment = "連番 - 連番1～300" },
                                new NormalFieldDefinition { Position = 4, ColumnName = "blood_registration_number", Length = 10, IsPrimaryKey = false, Comment = "血統登録番号 - 生年4桁＋品種1桁＋数字5桁" },
                                new NormalFieldDefinition { Position = 14, ColumnName = "horse_name", Length = 36, IsPrimaryKey = false, Comment = "馬名 - 全角18文字" },
                                new NormalFieldDefinition { Position = 50, ColumnName = "horse_symbol_code", Length = 2, IsPrimaryKey = false, Comment = "馬記号コード - <コード表 2204.馬記号コード>参照" },
                                new NormalFieldDefinition { Position = 52, ColumnName = "sex_code", Length = 1, IsPrimaryKey = false, Comment = "性別コード - <コード表 2202.性別コード>参照" },
                                new NormalFieldDefinition { Position = 53, ColumnName = "trainer_area_code", Length = 1, IsPrimaryKey = false, Comment = "調教師東西所属コード - <コード表 2301.東西所属コード>参照" },
                                new NormalFieldDefinition { Position = 54, ColumnName = "trainer_code", Length = 5, IsPrimaryKey = false, Comment = "調教師コード - 調教師マスタへリンク" },
                                new NormalFieldDefinition { Position = 59, ColumnName = "trainer_name_short", Length = 8, IsPrimaryKey = false, Comment = "調教師名略称 - 全角4文字" },
                                new NormalFieldDefinition { Position = 67, ColumnName = "burden_weight", Length = 3, IsPrimaryKey = false, Comment = "負担重量 - 単位:0.1kg ハンデキャップレースについては月曜以降に設定" },
                                new NormalFieldDefinition { Position = 70, ColumnName = "exchange_kubun", Length = 1, IsPrimaryKey = false, Comment = "交流区分 - 中央交流登録馬の場合に設定 0:初期値 1:地方馬 2:外国馬" }
                            }
                        }
                    },
                    creationDateField = new FieldDefinition { Position = 4, Length = 8 }
                });

            // レース詳細レコード定義
            recordDefinitions.Add(
                new RecordDefinition
                {
                    TableName = "race_detail",
                    RecordTypeId = "RA",
                    Comment = "レース詳細レコード - 出走馬名表から成績まで、レースの詳細情報",
                    Fields = new List<FieldDefinition>
                    {
                        new NormalFieldDefinition { Position = 1, ColumnName = "record_type_id", Length = 2, IsPrimaryKey = false, Comment = "レコード種別ID - 'RA' をセット" },
                        new NormalFieldDefinition { Position = 3, ColumnName = "data_kubun", Length = 1, IsPrimaryKey = false, Comment = "データ区分 - 1:出走馬名表(木曜) 2:出馬表(金･土曜) 3:速報成績(3着まで確定) 4:速報成績(5着まで確定) 5:速報成績(全馬着順確定) 6:速報成績(全馬着順+コーナ通過順) 7:成績(月曜) A:地方競馬 B:海外国際レース 9:レース中止 0:該当レコード削除" },
                        new NormalFieldDefinition { Position = 12, ColumnName = "kaisai_year", Length = 4, IsPrimaryKey = true, Comment = "開催年 - 該当レース施行年 西暦4桁 yyyy形式" },
                        new NormalFieldDefinition { Position = 16, ColumnName = "kaisai_month_day", Length = 4, IsPrimaryKey = true, Comment = "開催月日 - 該当レース施行月日 各2桁 mmdd形式" },
                        new NormalFieldDefinition { Position = 20, ColumnName = "track_code", Length = 2, IsPrimaryKey = true, Comment = "競馬場コード - 該当レース施行競馬場 <コード表 2001.競馬場コード>参照" },
                        new NormalFieldDefinition { Position = 22, ColumnName = "kaisai_kai", Length = 2, IsPrimaryKey = true, Comment = "開催回[第N回] - 該当レース施行回 その競馬場でその年の何回目の開催かを示す" },
                        new NormalFieldDefinition { Position = 24, ColumnName = "kaisai_day", Length = 2, IsPrimaryKey = true, Comment = "開催日目[N日目] - そのレース施行日目 そのレース施行回で何日目の開催かを示す" },
                        new NormalFieldDefinition { Position = 26, ColumnName = "race_number", Length = 2, IsPrimaryKey = true, Comment = "レース番号 - 該当レース番号 海外国際レースなどでレース番号情報がない場合は任意に連番を設定" },
                        new NormalFieldDefinition { Position = 28, ColumnName = "day_of_week_code", Length = 1, IsPrimaryKey = false, Comment = "曜日コード - 該当レース施行曜日 <コード表 2002.曜日コード>参照" },
                        new NormalFieldDefinition { Position = 29, ColumnName = "special_race_number", Length = 4, IsPrimaryKey = false, Comment = "特別競走番号 - 重賞レースのみ設定 原則的には過去の同一レースと一致する番号(多数例外有り)" },
                        new NormalFieldDefinition { Position = 33, ColumnName = "race_name_main", Length = 60, IsPrimaryKey = false, Comment = "競走名本題 - 全角30文字 レース名の本題" },
                        new NormalFieldDefinition { Position = 93, ColumnName = "race_name_sub", Length = 60, IsPrimaryKey = false, Comment = "競走名副題 - 全角30文字 レース名の副題（スポンサー名や記念名など）" },
                        new NormalFieldDefinition { Position = 153, ColumnName = "race_name_kakko", Length = 60, IsPrimaryKey = false, Comment = "競走名カッコ内 - 全角30文字 レースの条件やトライアル対象レース名、レース名通称など" },
                        new NormalFieldDefinition { Position = 213, ColumnName = "race_name_main_eng", Length = 120, IsPrimaryKey = false, Comment = "競走名本題欧字 - 半角120文字" },
                        new NormalFieldDefinition { Position = 333, ColumnName = "race_name_sub_eng", Length = 120, IsPrimaryKey = false, Comment = "競走名副題欧字 - 半角120文字" },
                        new NormalFieldDefinition { Position = 453, ColumnName = "race_name_kakko_eng", Length = 120, IsPrimaryKey = false, Comment = "競走名カッコ内欧字 - 半角120文字" },
                        new NormalFieldDefinition { Position = 573, ColumnName = "race_name_short10", Length = 20, IsPrimaryKey = false, Comment = "競走名略称10文字 - 全角10文字" },
                        new NormalFieldDefinition { Position = 593, ColumnName = "race_name_short6", Length = 12, IsPrimaryKey = false, Comment = "競走名略称6文字 - 全角6文字" },
                        new NormalFieldDefinition { Position = 605, ColumnName = "race_name_short3", Length = 6, IsPrimaryKey = false, Comment = "競走名略称3文字 - 全角3文字" },
                        new NormalFieldDefinition { Position = 611, ColumnName = "race_name_kubun", Length = 1, IsPrimaryKey = false, Comment = "競走名区分 - 重賞回次[第N回]を本題･副題･カッコ内のうちどれに設定すべきかを示す (0:初期値 1:本題 2:副題 3:カッコ内) 重賞のみ設定" },
                        new NormalFieldDefinition { Position = 612, ColumnName = "jyusho_kaiji", Length = 3, IsPrimaryKey = false, Comment = "重賞回次[第N回] - そのレースの重賞としての通算回数を示す" },
                        new NormalFieldDefinition { Position = 615, ColumnName = "grade_code", Length = 1, IsPrimaryKey = false, Comment = "グレードコード - <コード表 2003.グレードコード>参照" },
                        new NormalFieldDefinition { Position = 616, ColumnName = "grade_code_before", Length = 1, IsPrimaryKey = false, Comment = "変更前グレードコード - なんらかの理由により変更された場合のみ変更前の値を設定" },
                        new NormalFieldDefinition { Position = 617, ColumnName = "race_type_code", Length = 2, IsPrimaryKey = false, Comment = "競走種別コード - <コード表 2005.競走種別コード>参照" },
                        new NormalFieldDefinition { Position = 619, ColumnName = "race_symbol_code", Length = 3, IsPrimaryKey = false, Comment = "競走記号コード - <コード表 2006.競走記号コード>参照" },
                        new NormalFieldDefinition { Position = 622, ColumnName = "weight_type_code", Length = 1, IsPrimaryKey = false, Comment = "重量種別コード - <コード表 2008.重量種別コード>参照" },
                        new NormalFieldDefinition { Position = 623, ColumnName = "race_condition_2yo", Length = 3, IsPrimaryKey = false, Comment = "競走条件コード 2歳条件 - 2歳馬の競走条件 <コード表 2007.競走条件コード>参照" },
                        new NormalFieldDefinition { Position = 626, ColumnName = "race_condition_3yo", Length = 3, IsPrimaryKey = false, Comment = "競走条件コード 3歳条件 - 3歳馬の競走条件 <コード表 2007.競走条件コード>参照" },
                        new NormalFieldDefinition { Position = 629, ColumnName = "race_condition_4yo", Length = 3, IsPrimaryKey = false, Comment = "競走条件コード 4歳条件 - 4歳馬の競走条件 <コード表 2007.競走条件コード>参照" },
                        new NormalFieldDefinition { Position = 632, ColumnName = "race_condition_5yo_up", Length = 3, IsPrimaryKey = false, Comment = "競走条件コード 5歳以上条件 - 5歳以上馬の競走条件 <コード表 2007.競走条件コード>参照" },
                        new NormalFieldDefinition { Position = 635, ColumnName = "race_condition_youngest", Length = 3, IsPrimaryKey = false, Comment = "競走条件コード 最若年条件 - 出走可能な最も馬齢が若い馬に対する条件 <コード表 2007.競走条件コード>参照" },
                        new NormalFieldDefinition { Position = 638, ColumnName = "race_condition_name", Length = 60, IsPrimaryKey = false, Comment = "競走条件名称 - 全角30文字 地方競馬の場合のみ設定" },
                        new NormalFieldDefinition { Position = 698, ColumnName = "distance", Length = 4, IsPrimaryKey = false, Comment = "距離 - 単位:メートル" },
                        new NormalFieldDefinition { Position = 702, ColumnName = "distance_before", Length = 4, IsPrimaryKey = false, Comment = "変更前距離 - なんらかの理由により変更された場合のみ変更前の値を設定" },
                        new NormalFieldDefinition { Position = 706, ColumnName = "track_code_detail", Length = 2, IsPrimaryKey = false, Comment = "トラックコード - <コード表 2009.トラックコード>参照" },
                        new NormalFieldDefinition { Position = 708, ColumnName = "track_code_detail_before", Length = 2, IsPrimaryKey = false, Comment = "変更前トラックコード - なんらかの理由により変更された場合のみ変更前の値を設定" },
                        new NormalFieldDefinition { Position = 710, ColumnName = "course_kubun", Length = 2, IsPrimaryKey = false, Comment = "コース区分 - 半角2文字 使用するコースを設定 A～E を設定 2002年以前の東京競馬場はA1、A2も存在" },
                        new NormalFieldDefinition { Position = 712, ColumnName = "course_kubun_before", Length = 2, IsPrimaryKey = false, Comment = "変更前コース区分 - なんらかの理由により変更された場合のみ変更前の値を設定" },
                        new RepeatFieldDefinition
                        {
                            Position = 714,
                            TableName = "race_detail_prize",
                            RepeatCount = 7,
                            Length = 8,
                            Comment = "本賞金 - 単位:百円 1着～5着の本賞金 5着3同着まで考慮し繰返し7回",
                            Fields = new List<FieldDefinition>
                            {
                                new NormalFieldDefinition { Position = 1, ColumnName = "prize_money", Length = 8, IsPrimaryKey = false, Comment = "本賞金 - 単位:百円" }
                            }
                        },
                        new RepeatFieldDefinition
                        {
                            Position = 770,
                            TableName = "race_detail_prize_before",
                            RepeatCount = 5,
                            Length = 8,
                            Comment = "変更前本賞金 - 単位:百円 同着により本賞金の分配が変更された場合のみ変更前の値を設定",
                            Fields = new List<FieldDefinition>
                            {
                                new NormalFieldDefinition { Position = 1, ColumnName = "prize_money_before", Length = 8, IsPrimaryKey = false, Comment = "変更前本賞金 - 単位:百円" }
                            }
                        },
                        new RepeatFieldDefinition
                        {
                            Position = 810,
                            TableName = "race_detail_additional_prize",
                            RepeatCount = 5,
                            Length = 8,
                            Comment = "付加賞金 - 単位:百円 1着～3着の付加賞金 3着3同着まで考慮し繰返し5回",
                            Fields = new List<FieldDefinition>
                            {
                                new NormalFieldDefinition { Position = 1, ColumnName = "additional_prize_money", Length = 8, IsPrimaryKey = false, Comment = "付加賞金 - 単位:百円" }
                            }
                        },
                        new RepeatFieldDefinition
                        {
                            Position = 850,
                            TableName = "race_detail_additional_prize_before",
                            RepeatCount = 3,
                            Length = 8,
                            Comment = "変更前付加賞金 - 単位:百円 同着により付加賞金の分配が変更された場合のみ変更前の値を設定",
                            Fields = new List<FieldDefinition>
                            {
                                new NormalFieldDefinition { Position = 1, ColumnName = "additional_prize_money_before", Length = 8, IsPrimaryKey = false, Comment = "変更前付加賞金 - 単位:百円" }
                            }
                        },
                        new NormalFieldDefinition { Position = 874, ColumnName = "start_time", Length = 4, IsPrimaryKey = false, Comment = "発走時刻 - 時分各2桁 hhmm形式" },
                        new NormalFieldDefinition { Position = 878, ColumnName = "start_time_before", Length = 4, IsPrimaryKey = false, Comment = "変更前発走時刻 - なんらかの理由により変更された場合のみ変更前の値を設定" },
                        new NormalFieldDefinition { Position = 882, ColumnName = "registration_count", Length = 2, IsPrimaryKey = false, Comment = "登録頭数 - 出走馬名表時点:出走馬名表時点での登録頭数 出馬表発表時点:出馬表発表時の登録頭数(出馬表発表前に取消した馬を除いた頭数)" },
                        new NormalFieldDefinition { Position = 884, ColumnName = "start_count", Length = 2, IsPrimaryKey = false, Comment = "出走頭数 - 実際にレースに出走した頭数 (登録頭数から出走取消と競走除外･発走除外を除いた頭数)" },
                        new NormalFieldDefinition { Position = 886, ColumnName = "finish_count", Length = 2, IsPrimaryKey = false, Comment = "入線頭数 - 出走頭数から競走中止を除いた頭数" },
                        new NormalFieldDefinition { Position = 888, ColumnName = "weather_code", Length = 1, IsPrimaryKey = false, Comment = "天候コード - <コード表 2011.天候コード>参照" },
                        new NormalFieldDefinition { Position = 889, ColumnName = "turf_condition_code", Length = 1, IsPrimaryKey = false, Comment = "芝馬場状態コード - <コード表 2010.馬場状態コード>参照" },
                        new NormalFieldDefinition { Position = 890, ColumnName = "dirt_condition_code", Length = 1, IsPrimaryKey = false, Comment = "ダート馬場状態コード - <コード表 2010.馬場状態コード>参照" },
                        new RepeatFieldDefinition
                        {
                            Position = 891,
                            TableName = "race_detail_lap_time",
                            RepeatCount = 25,
                            Length = 3,
                            Comment = "ラップタイム - 99.9秒 平地競走のみ設定 1ハロン(200メートル)毎地点での先頭馬ラップタイム",
                            Fields = new List<FieldDefinition>
                            {
                                new NormalFieldDefinition { Position = 1, ColumnName = "lap_time", Length = 3, IsPrimaryKey = false, Comment = "ラップタイム - 99.9秒 1ハロン毎の先頭馬タイム" }
                            }
                        },
                        new NormalFieldDefinition { Position = 966, ColumnName = "obstacle_mile_time", Length = 4, IsPrimaryKey = false, Comment = "障害マイルタイム - 障害競走のみ設定 先頭馬の1マイル(1600メートル)通過タイム" },
                        new NormalFieldDefinition { Position = 970, ColumnName = "first_3furlong", Length = 3, IsPrimaryKey = false, Comment = "前3ハロン - 99.9秒 平地競走のみ設定 ラップタイム前半3ハロンの合計" },
                        new NormalFieldDefinition { Position = 973, ColumnName = "first_4furlong", Length = 3, IsPrimaryKey = false, Comment = "前4ハロン - 99.9秒 平地競走のみ設定 ラップタイム前半4ハロンの合計" },
                        new NormalFieldDefinition { Position = 976, ColumnName = "last_3furlong", Length = 3, IsPrimaryKey = false, Comment = "後3ハロン - 99.9秒 ラップタイム後半3ハロンの合計" },
                        new NormalFieldDefinition { Position = 979, ColumnName = "last_4furlong", Length = 3, IsPrimaryKey = false, Comment = "後4ハロン - 99.9秒 ラップタイム後半4ハロンの合計" },
                        new RepeatFieldDefinition
                        {
                            Position = 982,
                            TableName = "race_detail_corner_passing",
                            RepeatCount = 4,
                            Length = 72,
                            Comment = "コーナー通過順位 - 各コーナーでの通過順位情報",
                            Fields = new List<FieldDefinition>
                            {
                                new NormalFieldDefinition { Position = 1, ColumnName = "corner", Length = 1, IsPrimaryKey = false, Comment = "コーナー - 1:1コーナー 2:2コーナー 3:3コーナー 4:4コーナー" },
                                new NormalFieldDefinition { Position = 2, ColumnName = "lap_count", Length = 1, IsPrimaryKey = false, Comment = "周回数 - 1:1周 2:2周 3:3周" },
                                new NormalFieldDefinition { Position = 3, ColumnName = "passing_order", Length = 70, IsPrimaryKey = false, Comment = "各通過順位 - 順位を先頭内側から設定 ():集団 =:大差 -:小差 *:先頭集団のうちで先頭の馬番" }
                            }
                        },
                        new NormalFieldDefinition { Position = 1270, ColumnName = "record_update_kubun", Length = 1, IsPrimaryKey = false, Comment = "レコード更新区分 - 0:初期値 1:基準タイムとなったレース 2:コースレコードを更新したレース" }
                    },
                    creationDateField = new FieldDefinition { Position = 4, Length = 8 }
                });

            // 馬毎レース情報レコード定義
            recordDefinitions.Add(
                new RecordDefinition
                {
                    TableName = "horse_race_info",
                    RecordTypeId = "SE",
                    Comment = "馬毎レース情報レコード - 各馬のレース毎の詳細情報",
                    Fields = new List<FieldDefinition>
                    {
                        new NormalFieldDefinition { Position = 1, ColumnName = "record_type_id", Length = 2, IsPrimaryKey = false, Comment = "レコード種別ID - 'SE' をセット" },
                        new NormalFieldDefinition { Position = 3, ColumnName = "data_kubun", Length = 1, IsPrimaryKey = false, Comment = "データ区分 - 1:出走馬名表(木曜) 2:出馬表(金･土曜) 3:速報成績(3着まで確定) 4:速報成績(5着まで確定) 5:速報成績(全馬着順確定) 6:速報成績(全馬着順+コーナ通過順) 7:成績(月曜) A:地方競馬 B:海外国際レース 9:レース中止 0:該当レコード削除" },
                        new NormalFieldDefinition { Position = 12, ColumnName = "kaisai_year", Length = 4, IsPrimaryKey = true, Comment = "開催年 - 該当レース施行年 西暦4桁 yyyy形式" },
                        new NormalFieldDefinition { Position = 16, ColumnName = "kaisai_month_day", Length = 4, IsPrimaryKey = true, Comment = "開催月日 - 該当レース施行月日 各2桁 mmdd形式" },
                        new NormalFieldDefinition { Position = 20, ColumnName = "track_code", Length = 2, IsPrimaryKey = true, Comment = "競馬場コード - 該当レース施行競馬場 <コード表 2001.競馬場コード>参照" },
                        new NormalFieldDefinition { Position = 22, ColumnName = "kaisai_kai", Length = 2, IsPrimaryKey = true, Comment = "開催回[第N回] - 該当レース施行回 その競馬場でその年の何回目の開催かを示す" },
                        new NormalFieldDefinition { Position = 24, ColumnName = "kaisai_day", Length = 2, IsPrimaryKey = true, Comment = "開催日目[N日目] - そのレース施行回で何日目の開催かを示す" },
                        new NormalFieldDefinition { Position = 26, ColumnName = "race_number", Length = 2, IsPrimaryKey = true, Comment = "レース番号 - 該当レース番号" },
                        new NormalFieldDefinition { Position = 28, ColumnName = "frame_number", Length = 1, IsPrimaryKey = false, Comment = "枠番" },
                        new NormalFieldDefinition { Position = 29, ColumnName = "horse_number", Length = 2, IsPrimaryKey = true, Comment = "馬番 - 特定のレース及び海外レースについては、特記事項を参照" },
                        new NormalFieldDefinition { Position = 31, ColumnName = "blood_registration_number", Length = 10, IsPrimaryKey = false, Comment = "血統登録番号 - 生年(西暦)4桁＋品種1桁<コード表2201.品種コード>参照＋数字5桁" },
                        new NormalFieldDefinition { Position = 41, ColumnName = "horse_name", Length = 36, IsPrimaryKey = false, Comment = "馬名 - 通常全角18文字。海外レースにおける外国馬の場合のみ全角と半角が混在" },
                        new NormalFieldDefinition { Position = 77, ColumnName = "horse_symbol_code", Length = 2, IsPrimaryKey = false, Comment = "馬記号コード - <コード表 2204.馬記号コード>参照" },
                        new NormalFieldDefinition { Position = 79, ColumnName = "sex_code", Length = 1, IsPrimaryKey = false, Comment = "性別コード - <コード表 2202.性別コード>参照" },
                        new NormalFieldDefinition { Position = 80, ColumnName = "breed_code", Length = 1, IsPrimaryKey = false, Comment = "品種コード - <コード表 2201.品種コード>参照" },
                        new NormalFieldDefinition { Position = 81, ColumnName = "coat_color_code", Length = 2, IsPrimaryKey = false, Comment = "毛色コード - <コード表 2203.毛色コード>参照" },
                        new NormalFieldDefinition { Position = 83, ColumnName = "horse_age", Length = 2, IsPrimaryKey = false, Comment = "馬齢 - 出走当時の馬齢 (注)2000年以前は数え年表記 2001年以降は満年齢表記" },
                        new NormalFieldDefinition { Position = 85, ColumnName = "trainer_area_code", Length = 1, IsPrimaryKey = false, Comment = "東西所属コード - <コード表 2301.東西所属コード>参照" },
                        new NormalFieldDefinition { Position = 86, ColumnName = "trainer_code", Length = 5, IsPrimaryKey = false, Comment = "調教師コード - 調教師マスタへリンク" },
                        new NormalFieldDefinition { Position = 91, ColumnName = "trainer_name_short", Length = 8, IsPrimaryKey = false, Comment = "調教師名略称 - 全角4文字" },
                        new NormalFieldDefinition { Position = 99, ColumnName = "owner_code", Length = 6, IsPrimaryKey = false, Comment = "馬主コード - 馬主マスタへリンク" },
                        new NormalFieldDefinition { Position = 105, ColumnName = "owner_name_no_corp", Length = 64, IsPrimaryKey = false, Comment = "馬主名(法人格無) - 全角32文字 ～ 半角64文字 (全角と半角が混在) 株式会社、有限会社などの法人格を示す文字列が頭もしくは末尾にある場合にそれを削除したものを設定。また、外国馬主の場合は、馬主マスタの8.馬主名欧字の頭64バイトを設定" },
                        new NormalFieldDefinition { Position = 169, ColumnName = "racing_color", Length = 60, IsPrimaryKey = false, Comment = "服色標示 - 全角30文字 馬主毎に指定される騎手の勝負服の色・模様を示す (レーシングプログラムに記載されているもの) (例)\"水色，赤山形一本輪，水色袖\"" },
                        new NormalFieldDefinition { Position = 229, ColumnName = "reserve1", Length = 60, IsPrimaryKey = false, Comment = "予備" },
                        new NormalFieldDefinition { Position = 289, ColumnName = "burden_weight", Length = 3, IsPrimaryKey = false, Comment = "負担重量 - 単位0.1kg" },
                        new NormalFieldDefinition { Position = 292, ColumnName = "burden_weight_before", Length = 3, IsPrimaryKey = false, Comment = "変更前負担重量 - なんらかの理由により変更された場合のみ変更前の値を設定" },
                        new NormalFieldDefinition { Position = 295, ColumnName = "blinker_use_kubun", Length = 1, IsPrimaryKey = false, Comment = "ブリンカー使用区分 - 0:未使用 1:使用" },
                        new NormalFieldDefinition { Position = 296, ColumnName = "reserve2", Length = 1, IsPrimaryKey = false, Comment = "予備" },
                        new NormalFieldDefinition { Position = 297, ColumnName = "jockey_code", Length = 5, IsPrimaryKey = false, Comment = "騎手コード - 騎手マスタへリンク" },
                        new NormalFieldDefinition { Position = 302, ColumnName = "jockey_code_before", Length = 5, IsPrimaryKey = false, Comment = "変更前騎手コード - なんらかの理由により変更された場合のみ変更前の値を設定" },
                        new NormalFieldDefinition { Position = 307, ColumnName = "jockey_name_short", Length = 8, IsPrimaryKey = false, Comment = "騎手名略称 - 全角4文字" },
                        new NormalFieldDefinition { Position = 315, ColumnName = "jockey_name_short_before", Length = 8, IsPrimaryKey = false, Comment = "変更前騎手名略称 - なんらかの理由により変更された場合のみ変更前の値を設定" },
                        new NormalFieldDefinition { Position = 323, ColumnName = "jockey_apprentice_code", Length = 1, IsPrimaryKey = false, Comment = "騎手見習コード - <コード表 2303.騎手見習コード>参照" },
                        new NormalFieldDefinition { Position = 324, ColumnName = "jockey_apprentice_code_before", Length = 1, IsPrimaryKey = false, Comment = "変更前騎手見習コード - なんらかの理由により変更された場合のみ変更前の値を設定" },
                        new NormalFieldDefinition { Position = 325, ColumnName = "horse_weight", Length = 3, IsPrimaryKey = false, Comment = "馬体重 - 単位:kg 002Kg～998Kgまでが有効値 999:今走計量不能 000:出走取消" },
                        new NormalFieldDefinition { Position = 328, ColumnName = "weight_change_sign", Length = 1, IsPrimaryKey = false, Comment = "増減符号 - +:増加 -:減少 スペース:その他" },
                        new NormalFieldDefinition { Position = 329, ColumnName = "weight_change", Length = 3, IsPrimaryKey = false, Comment = "増減差 - 単位:kg 001Kg～998Kgまでが有効値 999:計量不能 000:前差なし スペース:初出走、ただし出走取消の場合もスペースを設定。地方馬については初出走かつ計量不能の場合でも\"999\"を設定。" },
                        new NormalFieldDefinition { Position = 332, ColumnName = "abnormal_kubun_code", Length = 1, IsPrimaryKey = false, Comment = "異常区分コード - <コード表 2101.異常区分コード>参照" },
                        new NormalFieldDefinition { Position = 333, ColumnName = "entry_order", Length = 2, IsPrimaryKey = false, Comment = "入線順位 - 失格、降着確定前の順位" },
                        new NormalFieldDefinition { Position = 335, ColumnName = "final_order", Length = 2, IsPrimaryKey = false, Comment = "確定着順 - 失格、降着時は入線順位と異なる" },
                        new NormalFieldDefinition { Position = 337, ColumnName = "dead_heat_kubun", Length = 1, IsPrimaryKey = false, Comment = "同着区分 - 0:同着馬なし 1:同着馬あり" },
                        new NormalFieldDefinition { Position = 338, ColumnName = "dead_heat_count", Length = 1, IsPrimaryKey = false, Comment = "同着頭数 - 0:初期値 1:自身以外に同着1頭 2:自身以外に同着2頭" },
                        new NormalFieldDefinition { Position = 339, ColumnName = "finish_time", Length = 4, IsPrimaryKey = false, Comment = "走破タイム - 9分99秒9で設定" },
                        new NormalFieldDefinition { Position = 343, ColumnName = "time_diff_code", Length = 3, IsPrimaryKey = false, Comment = "着差コード - 前馬との着差 <コード表 2102.着差コード>参照" },
                        new NormalFieldDefinition { Position = 346, ColumnName = "time_diff_code_plus", Length = 3, IsPrimaryKey = false, Comment = "＋着差コード - 前馬が失格、降着発生時に設定 前馬と前馬の前馬との着差" },
                        new NormalFieldDefinition { Position = 349, ColumnName = "time_diff_code_plus2", Length = 3, IsPrimaryKey = false, Comment = "＋＋着差コード - 前馬2頭が失格、降着発生時に設定" },
                        new NormalFieldDefinition { Position = 352, ColumnName = "corner1_order", Length = 2, IsPrimaryKey = false, Comment = "1コーナーでの順位" },
                        new NormalFieldDefinition { Position = 354, ColumnName = "corner2_order", Length = 2, IsPrimaryKey = false, Comment = "2コーナーでの順位" },
                        new NormalFieldDefinition { Position = 356, ColumnName = "corner3_order", Length = 2, IsPrimaryKey = false, Comment = "3コーナーでの順位" },
                        new NormalFieldDefinition { Position = 358, ColumnName = "corner4_order", Length = 2, IsPrimaryKey = false, Comment = "4コーナーでの順位" },
                        new NormalFieldDefinition { Position = 360, ColumnName = "win_odds", Length = 4, IsPrimaryKey = false, Comment = "単勝オッズ - 999.9倍で設定 出走取消し等は初期値を設定" },
                        new NormalFieldDefinition { Position = 364, ColumnName = "win_popularity", Length = 2, IsPrimaryKey = false, Comment = "単勝人気順 - 出走取消し等は初期値を設定" },
                        new NormalFieldDefinition { Position = 366, ColumnName = "main_prize_money", Length = 8, IsPrimaryKey = false, Comment = "獲得本賞金 - 単位:百円 該当レースで獲得した本賞金" },
                        new NormalFieldDefinition { Position = 374, ColumnName = "additional_prize_money", Length = 8, IsPrimaryKey = false, Comment = "獲得付加賞金 - 単位:百円 該当レースで獲得した付加賞金" },
                        new NormalFieldDefinition { Position = 382, ColumnName = "reserve3", Length = 3, IsPrimaryKey = false, Comment = "予備" },
                        new NormalFieldDefinition { Position = 385, ColumnName = "reserve4", Length = 3, IsPrimaryKey = false, Comment = "予備" },
                        new NormalFieldDefinition { Position = 388, ColumnName = "last_4furlong_time", Length = 3, IsPrimaryKey = false, Comment = "後4ハロンタイム - 単位:99.9秒 出走取消･競走除外･発走除外･競走中止･タイムオーバーの場合は\"999\"を設定 基本的には後3ハロンのみ設定(後4ハロンは初期値) ただし、過去分のデータは後4ハロンが設定されているものもある(その場合は後3ハロンが初期値) 障害レースの場合は後3ハロンに該当馬の当該レースでの1F平均タイムを設定(後4ハロンは初期値)" },
                        new NormalFieldDefinition { Position = 391, ColumnName = "last_3furlong_time", Length = 3, IsPrimaryKey = false, Comment = "後3ハロンタイム" },
                        new RepeatFieldDefinition
                        {
                            Position = 394,
                            TableName = "horse_race_info_rival",
                            RepeatCount = 3,
                            Length = 46,
                            Comment = "1着馬(相手馬)情報 - 同着を考慮して繰返し3回 自身が1着の場合は2着馬を設定",
                            Fields = new List<FieldDefinition>
                            {
                                new NormalFieldDefinition { Position = 1, ColumnName = "rival_blood_registration_number", Length = 10, IsPrimaryKey = false, Comment = "血統登録番号 - 生年(西暦)4桁＋品種1桁<コード表2201.品種コード>参照＋数字5桁" },
                                new NormalFieldDefinition { Position = 11, ColumnName = "rival_horse_name", Length = 36, IsPrimaryKey = false, Comment = "馬名 - 通常全角18文字。海外レースにおける外国馬の場合のみ全角と半角が混在。" }
                            }
                        },
                        new NormalFieldDefinition { Position = 532, ColumnName = "time_difference", Length = 4, IsPrimaryKey = false, Comment = "タイム差 - 1着馬とのタイム差を設定(自身が1着の場合は2着馬を設定) 符号(+または-)+99秒9 符号は1着:-, 2着以下:+ 出走取消･競走除外･発走除外･競走中止の場合は\"9999\"を設定" },
                        new NormalFieldDefinition { Position = 536, ColumnName = "record_update_kubun", Length = 1, IsPrimaryKey = false, Comment = "レコード更新区分 - 0:初期値 1:基準タイムとなったレース 2:コースレコードを更新したレース" },
                        new NormalFieldDefinition { Position = 537, ColumnName = "mining_kubun", Length = 1, IsPrimaryKey = false, Comment = "マイニング区分 - 1:前日 2:当日 3:直前 ただし、確定成績登録時に3:直前のみ設定" },
                        new NormalFieldDefinition { Position = 538, ColumnName = "mining_predicted_time", Length = 5, IsPrimaryKey = false, Comment = "マイニング予想走破タイム - 9分99秒99で設定" },
                        new NormalFieldDefinition { Position = 543, ColumnName = "mining_error_plus", Length = 4, IsPrimaryKey = false, Comment = "マイニング予想誤差(信頼度)＋ - 99秒99で設定 予想タイムの＋誤差を設定(＋方向の誤差。予想走破タイムに対して早くなる方向。予想走破タイムからマイナスする。)" },
                        new NormalFieldDefinition { Position = 547, ColumnName = "mining_error_minus", Length = 4, IsPrimaryKey = false, Comment = "マイニング予想誤差(信頼度)－ - 99秒99で設定 予想タイムの－誤差を設定(－方向の誤差。予想走破タイムに対して遅くなる方向。予想走破タイムにプラスする。)" },
                        new NormalFieldDefinition { Position = 551, ColumnName = "mining_predicted_order", Length = 2, IsPrimaryKey = false, Comment = "マイニング予想順位 - 01～18位を設定" },
                        new NormalFieldDefinition { Position = 553, ColumnName = "running_style_judgment", Length = 1, IsPrimaryKey = false, Comment = "今回レース脚質判定 - 1:逃 2:先 3:差 4:追 0:初期値" }
                    },
                    creationDateField = new FieldDefinition { Position = 4, Length = 8 }
                });

            // 払戻レコード定義
            recordDefinitions.Add(
                new RecordDefinition
                {
                    TableName = "payout",
                    RecordTypeId = "HR",
                    Comment = "払戻レコード - レースの払戻金情報",
                    Fields = new List<FieldDefinition>
                    {
                        new NormalFieldDefinition { Position = 1, ColumnName = "record_type_id", Length = 2, IsPrimaryKey = false, Comment = "レコード種別ID - 'HR' をセット" },
                        new NormalFieldDefinition { Position = 3, ColumnName = "data_kubun", Length = 1, IsPrimaryKey = false, Comment = "データ区分 - 1:速報成績(払戻金確定) 2:成績(月曜) 9:レース中止 0:該当レコード削除" },
                        new NormalFieldDefinition { Position = 12, ColumnName = "kaisai_year", Length = 4, IsPrimaryKey = true, Comment = "開催年 - 該当レース施行年 西暦4桁 yyyy形式" },
                        new NormalFieldDefinition { Position = 16, ColumnName = "kaisai_month_day", Length = 4, IsPrimaryKey = true, Comment = "開催月日 - 該当レース施行月日 各2桁 mmdd形式" },
                        new NormalFieldDefinition { Position = 20, ColumnName = "track_code", Length = 2, IsPrimaryKey = true, Comment = "競馬場コード - 該当レース施行競馬場 <コード表 2001.競馬場コード>参照" },
                        new NormalFieldDefinition { Position = 22, ColumnName = "kaisai_kai", Length = 2, IsPrimaryKey = true, Comment = "開催回[第N回] - 該当レース施行回 その競馬場でその年の何回目の開催かを示す" },
                        new NormalFieldDefinition { Position = 24, ColumnName = "kaisai_day", Length = 2, IsPrimaryKey = true, Comment = "開催日目[N日目] - そのレース施行回で何日目の開催かを示す" },
                        new NormalFieldDefinition { Position = 26, ColumnName = "race_number", Length = 2, IsPrimaryKey = true, Comment = "レース番号 - 該当レース番号" },
                        new NormalFieldDefinition { Position = 28, ColumnName = "registration_count", Length = 2, IsPrimaryKey = false, Comment = "登録頭数 - 出馬表発表時の登録頭数" },
                        new NormalFieldDefinition { Position = 30, ColumnName = "start_count", Length = 2, IsPrimaryKey = false, Comment = "出走頭数 - 登録頭数から出走取消と競走除外･発走除外を除いた頭数" },
                        new NormalFieldDefinition { Position = 32, ColumnName = "not_established_flag_tansho", Length = 1, IsPrimaryKey = false, Comment = "不成立フラグ 単勝 - 0:不成立なし 1:不成立あり" },
                        new NormalFieldDefinition { Position = 33, ColumnName = "not_established_flag_fukusho", Length = 1, IsPrimaryKey = false, Comment = "不成立フラグ 複勝 - 0:不成立なし 1:不成立あり" },
                        new NormalFieldDefinition { Position = 34, ColumnName = "not_established_flag_wakuren", Length = 1, IsPrimaryKey = false, Comment = "不成立フラグ 枠連 - 0:不成立なし 1:不成立あり" },
                        new NormalFieldDefinition { Position = 35, ColumnName = "not_established_flag_umaren", Length = 1, IsPrimaryKey = false, Comment = "不成立フラグ 馬連 - 0:不成立なし 1:不成立あり" },
                        new NormalFieldDefinition { Position = 36, ColumnName = "not_established_flag_wide", Length = 1, IsPrimaryKey = false, Comment = "不成立フラグ ワイド - 0:不成立なし 1:不成立あり" },
                        new NormalFieldDefinition { Position = 37, ColumnName = "reserve1", Length = 1, IsPrimaryKey = false, Comment = "予備" },
                        new NormalFieldDefinition { Position = 38, ColumnName = "not_established_flag_umatan", Length = 1, IsPrimaryKey = false, Comment = "不成立フラグ 馬単 - 0:不成立なし 1:不成立あり" },
                        new NormalFieldDefinition { Position = 39, ColumnName = "not_established_flag_3renpuku", Length = 1, IsPrimaryKey = false, Comment = "不成立フラグ 3連複 - 0:不成立なし 1:不成立あり" },
                        new NormalFieldDefinition { Position = 40, ColumnName = "not_established_flag_3rentan", Length = 1, IsPrimaryKey = false, Comment = "不成立フラグ 3連単 - 0:不成立なし 1:不成立あり" },
                        new NormalFieldDefinition { Position = 41, ColumnName = "special_flag_tansho", Length = 1, IsPrimaryKey = false, Comment = "特払フラグ 単勝 - 0:特払なし 1:特払あり" },
                        new NormalFieldDefinition { Position = 42, ColumnName = "special_flag_fukusho", Length = 1, IsPrimaryKey = false, Comment = "特払フラグ 複勝 - 0:特払なし 1:特払あり" },
                        new NormalFieldDefinition { Position = 43, ColumnName = "special_flag_wakuren", Length = 1, IsPrimaryKey = false, Comment = "特払フラグ 枠連 - 0:特払なし 1:特払あり" },
                        new NormalFieldDefinition { Position = 44, ColumnName = "special_flag_umaren", Length = 1, IsPrimaryKey = false, Comment = "特払フラグ 馬連 - 0:特払なし 1:特払あり" },
                        new NormalFieldDefinition { Position = 45, ColumnName = "special_flag_wide", Length = 1, IsPrimaryKey = false, Comment = "特払フラグ ワイド - 0:特払なし 1:特払あり" },
                        new NormalFieldDefinition { Position = 46, ColumnName = "reserve2", Length = 1, IsPrimaryKey = false, Comment = "予備" },
                        new NormalFieldDefinition { Position = 47, ColumnName = "special_flag_umatan", Length = 1, IsPrimaryKey = false, Comment = "特払フラグ 馬単 - 0:特払なし 1:特払あり" },
                        new NormalFieldDefinition { Position = 48, ColumnName = "special_flag_3renpuku", Length = 1, IsPrimaryKey = false, Comment = "特払フラグ 3連複 - 0:特払なし 1:特払あり" },
                        new NormalFieldDefinition { Position = 49, ColumnName = "special_flag_3rentan", Length = 1, IsPrimaryKey = false, Comment = "特払フラグ 3連単 - 0:特払なし 1:特払あり" },
                        new NormalFieldDefinition { Position = 50, ColumnName = "refund_flag_tansho", Length = 1, IsPrimaryKey = false, Comment = "返還フラグ 単勝 - 0:返還なし 1:返還あり" },
                        new NormalFieldDefinition { Position = 51, ColumnName = "refund_flag_fukusho", Length = 1, IsPrimaryKey = false, Comment = "返還フラグ 複勝 - 0:返還なし 1:返還あり" },
                        new NormalFieldDefinition { Position = 52, ColumnName = "refund_flag_wakuren", Length = 1, IsPrimaryKey = false, Comment = "返還フラグ 枠連 - 0:返還なし 1:返還あり" },
                        new NormalFieldDefinition { Position = 53, ColumnName = "refund_flag_umaren", Length = 1, IsPrimaryKey = false, Comment = "返還フラグ 馬連 - 0:返還なし 1:返還あり" },
                        new NormalFieldDefinition { Position = 54, ColumnName = "refund_flag_wide", Length = 1, IsPrimaryKey = false, Comment = "返還フラグ ワイド - 0:返還なし 1:返還あり" },
                        new NormalFieldDefinition { Position = 55, ColumnName = "reserve3", Length = 1, IsPrimaryKey = false, Comment = "予備" },
                        new NormalFieldDefinition { Position = 56, ColumnName = "refund_flag_umatan", Length = 1, IsPrimaryKey = false, Comment = "返還フラグ 馬単 - 0:返還なし 1:返還あり" },
                        new NormalFieldDefinition { Position = 57, ColumnName = "refund_flag_3renpuku", Length = 1, IsPrimaryKey = false, Comment = "返還フラグ 3連複 - 0:返還なし 1:返還あり" },
                        new NormalFieldDefinition { Position = 58, ColumnName = "refund_flag_3rentan", Length = 1, IsPrimaryKey = false, Comment = "返還フラグ 3連単 - 0:返還なし 1:返還あり" },
                        new NormalFieldDefinition { Position = 59, ColumnName = "refund_horse_info", Length = 28, IsPrimaryKey = false, Comment = "返還馬番情報(馬番01～28) - 0:返還なし 1:返還あり 発売後取消しとなり返還対象となった馬番" },
                        new NormalFieldDefinition { Position = 87, ColumnName = "refund_frame_info", Length = 8, IsPrimaryKey = false, Comment = "返還枠番情報(枠番1～8) - 0:返還なし 1:返還あり 発売後取消しとなり返還対象となった枠番" },
                        new NormalFieldDefinition { Position = 95, ColumnName = "refund_same_frame_info", Length = 8, IsPrimaryKey = false, Comment = "返還同枠情報(枠番1～8) - 0:返還なし 1:返還あり 発売後取消しとなり返還対象となった同枠" },
                        new RepeatFieldDefinition
                        {
                            Position = 103,
                            TableName = "payout_tansho",
                            RepeatCount = 3,
                            Length = 13,
                            Comment = "単勝払戻 - 3同着まで考慮し繰返し3回",
                            Fields = new List<FieldDefinition>
                            {
                                new NormalFieldDefinition { Position = 1, ColumnName = "horse_number", Length = 2, IsPrimaryKey = false, Comment = "馬番 - 00:発売なし、特払、不成立" },
                                new NormalFieldDefinition { Position = 3, ColumnName = "payout_amount", Length = 9, IsPrimaryKey = false, Comment = "払戻金 - 特払、不成立の金額が入る" },
                                new NormalFieldDefinition { Position = 12, ColumnName = "popularity", Length = 2, IsPrimaryKey = false, Comment = "人気順" }
                            }
                        },
                        new RepeatFieldDefinition
                        {
                            Position = 142,
                            TableName = "payout_fukusho",
                            RepeatCount = 5,
                            Length = 13,
                            Comment = "複勝払戻 - 3同着まで考慮し繰返し5回",
                            Fields = new List<FieldDefinition>
                            {
                                new NormalFieldDefinition { Position = 1, ColumnName = "horse_number", Length = 2, IsPrimaryKey = false, Comment = "馬番 - 00:発売なし、特払、不成立" },
                                new NormalFieldDefinition { Position = 3, ColumnName = "payout_amount", Length = 9, IsPrimaryKey = false, Comment = "払戻金 - 特払、不成立の金額が入る" },
                                new NormalFieldDefinition { Position = 12, ColumnName = "popularity", Length = 2, IsPrimaryKey = false, Comment = "人気順" }
                            }
                        },
                        new RepeatFieldDefinition
                        {
                            Position = 207,
                            TableName = "payout_wakuren",
                            RepeatCount = 3,
                            Length = 13,
                            Comment = "枠連払戻 - 3同着まで考慮し繰返し3回",
                            Fields = new List<FieldDefinition>
                            {
                                new NormalFieldDefinition { Position = 1, ColumnName = "combination", Length = 2, IsPrimaryKey = false, Comment = "組番 - 00:発売なし、特払、不成立" },
                                new NormalFieldDefinition { Position = 3, ColumnName = "payout_amount", Length = 9, IsPrimaryKey = false, Comment = "払戻金 - 特払、不成立の金額が入る" },
                                new NormalFieldDefinition { Position = 12, ColumnName = "popularity", Length = 2, IsPrimaryKey = false, Comment = "人気順" }
                            }
                        },
                        new RepeatFieldDefinition
                        {
                            Position = 246,
                            TableName = "payout_umaren",
                            RepeatCount = 3,
                            Length = 16,
                            Comment = "馬連払戻 - 3同着まで考慮し繰返し3回",
                            Fields = new List<FieldDefinition>
                            {
                                new NormalFieldDefinition { Position = 1, ColumnName = "combination", Length = 4, IsPrimaryKey = false, Comment = "組番 - 0000:発売なし、特払、不成立" },
                                new NormalFieldDefinition { Position = 5, ColumnName = "payout_amount", Length = 9, IsPrimaryKey = false, Comment = "払戻金 - 特払、不成立の金額が入る" },
                                new NormalFieldDefinition { Position = 14, ColumnName = "popularity", Length = 3, IsPrimaryKey = false, Comment = "人気順" }
                            }
                        },
                        new RepeatFieldDefinition
                        {
                            Position = 294,
                            TableName = "payout_wide",
                            RepeatCount = 7,
                            Length = 16,
                            Comment = "ワイド払戻 - 3同着まで考慮し繰返し7回",
                            Fields = new List<FieldDefinition>
                            {
                                new NormalFieldDefinition { Position = 1, ColumnName = "combination", Length = 4, IsPrimaryKey = false, Comment = "組番 - 0000:発売なし、特払、不成立" },
                                new NormalFieldDefinition { Position = 5, ColumnName = "payout_amount", Length = 9, IsPrimaryKey = false, Comment = "払戻金 - 特払、不成立の金額が入る" },
                                new NormalFieldDefinition { Position = 14, ColumnName = "popularity", Length = 3, IsPrimaryKey = false, Comment = "人気順" }
                            }
                        },
                        new RepeatFieldDefinition
                        {
                            Position = 406,
                            TableName = "payout_reserve",
                            RepeatCount = 3,
                            Length = 16,
                            Comment = "予備",
                            Fields = new List<FieldDefinition>
                            {
                                new NormalFieldDefinition { Position = 1, ColumnName = "reserve1", Length = 4, IsPrimaryKey = false, Comment = "予備" },
                                new NormalFieldDefinition { Position = 5, ColumnName = "reserve2", Length = 9, IsPrimaryKey = false, Comment = "予備" },
                                new NormalFieldDefinition { Position = 14, ColumnName = "reserve3", Length = 3, IsPrimaryKey = false, Comment = "予備" }
                            }
                        },
                        new RepeatFieldDefinition
                        {
                            Position = 454,
                            TableName = "payout_umatan",
                            RepeatCount = 6,
                            Length = 16,
                            Comment = "馬単払戻 - 3同着まで考慮し繰返し6回",
                            Fields = new List<FieldDefinition>
                            {
                                new NormalFieldDefinition { Position = 1, ColumnName = "combination", Length = 4, IsPrimaryKey = false, Comment = "組番 - 0000:発売なし、特払、不成立" },
                                new NormalFieldDefinition { Position = 5, ColumnName = "payout_amount", Length = 9, IsPrimaryKey = false, Comment = "払戻金 - 特払、不成立の金額が入る" },
                                new NormalFieldDefinition { Position = 14, ColumnName = "popularity", Length = 3, IsPrimaryKey = false, Comment = "人気順" }
                            }
                        },
                        new RepeatFieldDefinition
                        {
                            Position = 550,
                            TableName = "payout_3renpuku",
                            RepeatCount = 3,
                            Length = 18,
                            Comment = "3連複払戻 - 3同着まで考慮し繰返し3回",
                            Fields = new List<FieldDefinition>
                            {
                                new NormalFieldDefinition { Position = 1, ColumnName = "combination", Length = 6, IsPrimaryKey = false, Comment = "組番 - 000000:発売なし、特払、不成立" },
                                new NormalFieldDefinition { Position = 7, ColumnName = "payout_amount", Length = 9, IsPrimaryKey = false, Comment = "払戻金 - 特払、不成立の金額が入る" },
                                new NormalFieldDefinition { Position = 16, ColumnName = "popularity", Length = 3, IsPrimaryKey = false, Comment = "人気順" }
                            }
                        },
                        new RepeatFieldDefinition
                        {
                            Position = 604,
                            TableName = "payout_3rentan",
                            RepeatCount = 6,
                            Length = 19,
                            Comment = "3連単払戻 - 3同着まで考慮し繰返し6回",
                            Fields = new List<FieldDefinition>
                            {
                                new NormalFieldDefinition { Position = 1, ColumnName = "combination", Length = 6, IsPrimaryKey = false, Comment = "組番 - 000000:発売なし、特払、不成立" },
                                new NormalFieldDefinition { Position = 7, ColumnName = "payout_amount", Length = 9, IsPrimaryKey = false, Comment = "払戻金 - 特払、不成立の金額が入る" },
                                new NormalFieldDefinition { Position = 16, ColumnName = "popularity", Length = 4, IsPrimaryKey = false, Comment = "人気順" }
                            }
                        }
                    },
                    creationDateField = new FieldDefinition { Position = 4, Length = 8 }
                });
        }

    }
}

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
            public string RecordTypeId { get; set; }
            public TableDefinition Table { get; set; }
            public FieldDefinition creationDateField { get; set; }
        }

        public class TableDefinition
        {
            public string Name { get; set; }
            public List<FieldDefinition> Fields { get; set; }
            public string Comment { get; set; }
        }

        public class FieldDefinition
        {
            public int Position { get; set; }
            public int Length { get; set; }
        }

        public class NormalFieldDefinition : FieldDefinition
        {
            public string Name { get; set; }
            public bool IsPrimaryKey { get; set; } = false;
            public string Comment { get; set; }
        }

        public class RepeatFieldDefinition : FieldDefinition
        {
            public int RepeatCount { get; set; }
            public TableDefinition Table { get; set; }
        }

        public class TableMetaData
        {
            public string TableName { get; set; }
            public List<ColumnMetaData> Columns { get; set; }
            public List<string> PrimaryKeys { get; set; }
            public string Comment { get; set; }
        }

        public class ColumnMetaData
        {
            public string Name { get; set; }
            public string DataType { get; set; }
            public int Length { get; set; }
            public bool IsPrimaryKey { get; set; }
            public string Comment { get; set; }
        }

        private string[] commandLineArgs;
        private string connectionString = "Host=localhost;Database=horgues3;Username=postgres;Password=postgres";
        private List<RecordDefinition> recordDefinitions = new List<RecordDefinition>();
        private Dictionary<string, Dictionary<string, Dictionary<string, Object>>> buffers;
        private List<TableMetaData> tableMetaData = new List<TableMetaData>();
        private int batchSize = 100000;

        public JVLinkForm(string[] args)
        {
            InitializeComponent();
            commandLineArgs = args;
        }

        private void JVLinkForm_Load(object sender, EventArgs e)
        {
            try
            {
                Console.WriteLine($"JVDataCollector started at {DateTime.Now:yyyy-MM-dd HH:mm:ss}");

                // レコード定義の作成
                CreateRecordDefinitions();
                CreateTableMetaData();
                InitializeBuffers();

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
            result = 0;
            int previousResult = -1;

            while (result < filesToDownload)
            {
                // 少し待ってからJVStatusを呼び出し
                System.Threading.Thread.Sleep(1000); // 1秒待機

                result = axJVLink1.JVStatus();
                if (result < 0)
                {
                    throw new InvalidOperationException($"JVStatus failed with error code: {result}");
                }

                // ダウンロード済みファイル数が変化したときのみコンソール出力
                if (result != previousResult)
                {
                    Console.WriteLine($"Downloaded files: {result}/{filesToDownload}");
                    previousResult = result;
                }
            }

            Console.WriteLine($"All files downloaded successfully.");

            // JVReadでデータを読み出し処理
            Console.WriteLine("Starting data read and processing...");
            int totalRecords = 0;
            int totalFiles = 0;

            using (var connection = new NpgsqlConnection(connectionString))
            {
                connection.Open();

                while (true)
                {
                    string buff;
                    string filename;
                    int size = 110000;

                    // JVReadでデータを読み込み
                    result = axJVLink1.JVRead(out buff, out size, out filename);

                    if (result == 0)
                    {
                        // 全ファイル読み込み終了
                        Console.WriteLine("All files read completed.");
                        break;
                    }
                    else if (result == -1)
                    {
                        // ファイル切り替わり
                        totalFiles++;
                        Console.WriteLine($"File completed: {filename.Trim()} (File {totalFiles})");
                        continue;
                    }
                    else if (result > 0)
                    {
                        // データが読み込まれた
                        totalRecords++;
                        var recordData = System.Text.Encoding.GetEncoding("Shift_JIS").GetBytes(buff);
                        ProcessRecord(recordData, connection);
                    }
                    else
                    {
                        // エラー
                        throw new InvalidOperationException($"JVRead failed with error code: {result}");
                    }
                }

                Console.WriteLine($"Data processing completed. Total records: {totalRecords}, Total files: {totalFiles}");

                // バッファに残っているデータをフラッシュ
                FlushAllBuffers(connection);

                // TODO: lastFileTimestamp をデータベースに保存する

            }

            axJVLink1.JVClose();
            Console.WriteLine($"JV data processing completed.");
        }

        private void ProcessRecord(byte[] recordData, NpgsqlConnection connection)
        {
            // レコード種別IDを取得
            string recordTypeId = System.Text.Encoding.GetEncoding("Shift_JIS").GetString(recordData, 0, 2);

            // 対応するレコード定義を検索
            var recordDefinition = recordDefinitions.FirstOrDefault(r => r.RecordTypeId == recordTypeId);
            if (recordDefinition == null)
            {
                // 対応するレコード定義がない場合はスキップ
                return;
            }

            string creationDate = System.Text.Encoding.GetEncoding("Shift_JIS").GetString(recordData, recordDefinition.creationDateField.Position - 1, recordDefinition.creationDateField.Length);

            ProcessTablesRecursive(recordData, recordDefinition.Table, creationDate, new Dictionary<string, Object>(), new List<string>(), connection);
        }

        private void ProcessTablesRecursive(byte[] recordData, TableDefinition tableDefinition, string creationDate, Dictionary<string, Object> parentPrimaryKeys, List<string> parentTableNames, NpgsqlConnection connection)
        {
            string fullTableName = parentTableNames.Count > 0
                ? string.Join("_", parentTableNames) + "_" + tableDefinition.Name
                : tableDefinition.Name;
            var record = new Dictionary<string, Object>();
            var primaryKeys = new Dictionary<string, Object>();

            // 親テーブルの主キーを継承
            foreach (var parentKey in parentPrimaryKeys)
            {
                record[parentKey.Key] = parentKey.Value;
                primaryKeys[parentKey.Key] = parentKey.Value;
            }
            
            // 各フィールドを抽出
            foreach (var field in tableDefinition.Fields)
            {
                if (field is NormalFieldDefinition normalField)
                {
                    string value = System.Text.Encoding.GetEncoding("Shift_JIS").GetString(recordData, normalField.Position - 1, normalField.Length);
                    record[normalField.Name] = value;
                    if (normalField.IsPrimaryKey)
                    {
                        primaryKeys[normalField.Name] = value;
                    }
                }
            }

            // creationDateを追加
            record["creation_date"] = creationDate;

            // 重複をチェックしてバッファに追加し、バッチサイズに達したらUPSERTする
            BufferRecord(fullTableName, record, primaryKeys, connection);

            // 繰り返しフィールドの処理
            foreach (var field in tableDefinition.Fields)
            {
                if (field is RepeatFieldDefinition repeatField)
                {
                    for (int i = 0; i < repeatField.RepeatCount; i++)
                    {
                        // データを抽出
                        int instanceStart = (repeatField.Position - 1) + (i * repeatField.Length);
                        byte[] instanceData = new byte[repeatField.Length];
                        Array.Copy(recordData, instanceStart, instanceData, 0, repeatField.Length);

                        // 繰り返しインデックスを追加
                        var childPrimaryKeys = new Dictionary<string, Object>(primaryKeys);
                        string indexColumnName = $"{repeatField.Table.Name}_index";
                        childPrimaryKeys[indexColumnName] = i;

                        // 子テーブルを処理
                        ProcessTablesRecursive(instanceData, repeatField.Table, creationDate, childPrimaryKeys, parentTableNames.Append(tableDefinition.Name).ToList(), connection);
                    }
                }
            }
        }

        private void BufferRecord(string tableName, Dictionary<string, Object> record, Dictionary<string, Object> primaryKeys, NpgsqlConnection connection)
        {
            var buffer = buffers[tableName];

            // Create composite primary key string
            string primaryKeyString = string.Join("|", primaryKeys.OrderBy(pk => pk.Key).Select(pk => $"{pk.Key}:{pk.Value}"));

            // Check for existing record using primary key
            if (buffer.ContainsKey(primaryKeyString))
            {
                // 既存レコードの作成日と比較
                string existingCreationDate = buffer[primaryKeyString]["creation_date"].ToString();
                string newCreationDate = record["creation_date"].ToString();

                if (string.Compare(newCreationDate, existingCreationDate) >= 0)
                {
                    // 新しいデータの場合、既存レコードを置き換え
                    buffer[primaryKeyString] = record;
                }
                // 古いデータの場合は何もしない
            }
            else
            {
                // 新規レコードの場合、バッファに追加
                buffer[primaryKeyString] = record;
            }

            // バッチサイズに達したらUPSERTを実行
            if (buffer.Count >= batchSize)
            {
                ExecuteUpsertBatch(tableName, connection);
            }
        }

        private void ExecuteUpsertBatch(string tableName, NpgsqlConnection connection)
        {
            var buffer = buffers[tableName];
            if (buffer.Count == 0) return;

            // テーブルのメタデータを取得
            var metadata = tableMetaData.FirstOrDefault(t => t.TableName == tableName);
            if (metadata == null)
            {
                throw new InvalidOperationException($"Table metadata not found for table: {tableName}");
            }

            if (metadata.PrimaryKeys.Count == 0)
            {
                throw new InvalidOperationException($"No primary keys defined for table: {tableName}");
            }

            // カラム名リストを作成（creation_dateを含む）
            var allColumns = metadata.Columns.Select(c => c.Name).ToList();
            var primaryKeyColumns = metadata.PrimaryKeys;
            var nonPrimaryKeyColumns = allColumns.Except(primaryKeyColumns).ToList();

            // VALUES句を構築
            var valuesList = new List<string>();

            foreach (var record in buffer.Values)
            {
                var valueParams = new List<string>();
                
                foreach (var column in allColumns)
                {
                    var value = record.ContainsKey(column) ? record[column] : null;
                    
                    // 値をSQL文字列として適切にエスケープ
                    string sqlValue;
                    if (value == null || value == DBNull.Value)
                    {
                        sqlValue = "NULL";
                    }
                    else if (value is string stringValue)
                    {
                        if (string.IsNullOrEmpty(stringValue))
                        {
                            sqlValue = "NULL";
                        }
                        else
                        {
                            // シングルクォートをエスケープ
                            sqlValue = $"'{stringValue.Replace("'", "''")}'";
                        }
                    }
                    else
                    {
                        // 数値や他の型の場合
                        sqlValue = value.ToString();
                    }
                    
                    valueParams.Add(sqlValue);
                }
                
                valuesList.Add($"({string.Join(", ", valueParams)})");
            }

            // INSERT文の構築
            var insertColumns = string.Join(", ", allColumns);
            var valuesClause = string.Join(", ", valuesList);
            
            // ON CONFLICT句の構築
            var conflictColumns = string.Join(", ", primaryKeyColumns);
            
            // UPDATE句の構築
            var updateClauses = nonPrimaryKeyColumns.Select(col => $"{col} = EXCLUDED.{col}");
            var updateClause = string.Join(", ", updateClauses);

            // UPSERT SQLの構築
            var sql = $@"
                INSERT INTO {tableName} ({insertColumns})
                VALUES {valuesClause}
                ON CONFLICT ({conflictColumns})
                DO UPDATE SET {updateClause}
                WHERE EXCLUDED.creation_date >= {tableName}.creation_date";

            using (var command = new NpgsqlCommand(sql, connection))
            {
                command.ExecuteNonQuery();
            }

            // バッファをクリア
            buffer.Clear();
        }

        private void FlushAllBuffers(NpgsqlConnection connection)
        {
            Console.WriteLine("Flushing remaining data in all buffers...");

            foreach (var tableName in buffers.Keys.ToList())
            {
                if (buffers[tableName].Count > 0)
                {
                    ExecuteUpsertBatch(tableName, connection);
                }
            }

            Console.WriteLine("All buffers flushed.");
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

        private void CreateTableMetaData()
        {
            Console.WriteLine("Creating table metadata...");

            tableMetaData.Clear();
            
            foreach (var recordDef in recordDefinitions)
            {
                CreateTableMetaDataRecursive(recordDef.Table, new List<NormalFieldDefinition>(), new List<string>());
            }

            Console.WriteLine($"Table metadata created successfully. Total tables: {tableMetaData.Count}");
        }

        private void CreateTableMetaDataRecursive(TableDefinition table, List<NormalFieldDefinition> parentPrimaryKeys, List<string> parentTableNames)
        {
            string fullTableName = parentTableNames.Count > 0
                ? string.Join("_", parentTableNames) + "_" + table.Name
                : table.Name;

            var metadata = new TableMetaData
            {
                TableName = fullTableName,
                Columns = new List<ColumnMetaData>(),
                PrimaryKeys = new List<string>(),
                Comment = table.Comment
            };

            // 親テーブルの主キーを継承
            foreach (var pk in parentPrimaryKeys)
            {
                metadata.Columns.Add(new ColumnMetaData
                {
                    Name = pk.Name,
                    DataType = "CHAR",
                    Length = pk.Length,
                    IsPrimaryKey = true,
                    Comment = pk.Comment
                });
                metadata.PrimaryKeys.Add(pk.Name);
            }

            // 繰り返し番号を追加
            if (parentTableNames.Count > 0)
            {
                string indexColumnName = $"{table.Name}_index";
                metadata.Columns.Add(new ColumnMetaData
                {
                    Name = indexColumnName,
                    DataType = "INTEGER",
                    Length = 0,
                    IsPrimaryKey = true,
                    Comment = "Repeat index"
                });
                metadata.PrimaryKeys.Add(indexColumnName);
            }

            var currentPrimaryKeys = new List<NormalFieldDefinition>(parentPrimaryKeys);

            foreach (var field in table.Fields)
            {
                if (field is NormalFieldDefinition normalField)
                {
                    metadata.Columns.Add(new ColumnMetaData
                    {
                        Name = normalField.Name,
                        DataType = "CHAR",
                        Length = normalField.Length,
                        IsPrimaryKey = normalField.IsPrimaryKey,
                        Comment = normalField.Comment
                    });
                    
                    if (normalField.IsPrimaryKey)
                    {
                        metadata.PrimaryKeys.Add(normalField.Name);
                        currentPrimaryKeys.Add(normalField);
                    }
                }
                else if (field is RepeatFieldDefinition repeatField)
                {
                    // 子テーブルを再帰的に処理
                    CreateTableMetaDataRecursive(repeatField.Table, currentPrimaryKeys, parentTableNames.Append(table.Name).ToList());
                }
            }

            // creation_dateカラムを追加
            metadata.Columns.Add(new ColumnMetaData
            {
                Name = "creation_date",
                DataType = "CHAR",
                Length = 8,
                IsPrimaryKey = false,
                Comment = "Data creation date"
            });

            tableMetaData.Add(metadata);
            Console.WriteLine($"  Table metadata created: {fullTableName}");
        }

        private void CreateTables()
        {
            Console.WriteLine("Creating database tables...");

            using (var connection = new NpgsqlConnection(connectionString))
            {
                connection.Open();

                foreach (var metadata in tableMetaData)
                {
                    // 既存のテーブルを削除
                    var dropTableSql = $"DROP TABLE IF EXISTS {metadata.TableName}";
                    using (var dropCommand = new NpgsqlCommand(dropTableSql, connection))
                    {
                        dropCommand.ExecuteNonQuery();
                    }

                    // カラム定義を作成
                    var columns = new List<string>();
                    foreach (var column in metadata.Columns)
                    {
                        string columnDef = column.DataType == "INTEGER" 
                            ? $"{column.Name} {column.DataType}"
                            : $"{column.Name} {column.DataType}({column.Length})";
                        columns.Add(columnDef);
                    }

                    // CREATE TABLE文を作成
                    var createTableSql = $"CREATE TABLE {metadata.TableName} ({string.Join(", ", columns)}";
                    if (metadata.PrimaryKeys.Count > 0)
                    {
                        createTableSql += $", PRIMARY KEY ({string.Join(", ", metadata.PrimaryKeys)})";
                    }
                    createTableSql += ")";

                    using (var command = new NpgsqlCommand(createTableSql, connection))
                    {
                        command.ExecuteNonQuery();
                    }

                    // テーブルコメントを追加
                    if (!string.IsNullOrEmpty(metadata.Comment))
                    {
                        var tableCommentSql = $"COMMENT ON TABLE {metadata.TableName} IS '{metadata.Comment.Replace("'", "''")}'";
                        using (var commentCommand = new NpgsqlCommand(tableCommentSql, connection))
                        {
                            commentCommand.ExecuteNonQuery();
                        }
                    }

                    // カラムコメントを追加
                    foreach (var column in metadata.Columns)
                    {
                        if (!string.IsNullOrEmpty(column.Comment))
                        {
                            var columnCommentSql = $"COMMENT ON COLUMN {metadata.TableName}.{column.Name} IS '{column.Comment.Replace("'", "''")}'";
                            using (var commentCommand = new NpgsqlCommand(columnCommentSql, connection))
                            {
                                commentCommand.ExecuteNonQuery();
                            }
                        }
                    }

                    Console.WriteLine($"Table '{metadata.TableName}' created.");
                }
            }

            Console.WriteLine("Database tables created successfully.");
        }

        private void ExecuteUpdate()
        {
            throw new NotImplementedException("Update functionality is not yet implemented");
        }

        private void InitializeBuffers()
        {
            Console.WriteLine("Initializing data buffers...");
            
            buffers = new Dictionary<string, Dictionary<string, Dictionary<string, Object>>>();
            foreach (var recordDef in recordDefinitions)
            {
                buffers[recordDef.Table.Name] = new Dictionary<string, Dictionary<string, Object>>();
                Console.WriteLine($"  Buffer initialized for table: {recordDef.Table.Name}");
                InitializeChildTableBuffers(recordDef.Table.Fields, new List<string> { recordDef.Table.Name });
            }
            
            Console.WriteLine("Data buffers initialization completed.");
        }

        private void InitializeChildTableBuffers(List<FieldDefinition> fields, List<string> parentTables)
        {
            foreach (var field in fields)
            {
                if (field is RepeatFieldDefinition repeatField)
                {
                    string fullTableName = string.Join("_", parentTables) + "_" + repeatField.Table.Name;
                    buffers[fullTableName] = new Dictionary<string, Dictionary<string, Object>>();
                    Console.WriteLine($"  Child buffer initialized for table: {fullTableName}");
                    InitializeChildTableBuffers(repeatField.Table.Fields, parentTables.Append(repeatField.Table.Name).ToList());
                }
            }
        }

        private void CreateRecordDefinitions()
        {
            // 特別登録馬レコード定義
            recordDefinitions.Add(
                new RecordDefinition
                {
                    RecordTypeId = "TK",
                    Table = new TableDefinition
                    {
                        Name = "special_registration",
                        Comment = "特別登録馬レコード: ハンデ発表前後の特別競走登録馬情報",
                        Fields = new List<FieldDefinition>
                        {
                            new NormalFieldDefinition { Position = 1, Name = "record_type_id", Length = 2, IsPrimaryKey = false, Comment = "レコード種別ID: 'TK' をセット" },
                            new NormalFieldDefinition { Position = 3, Name = "data_kubun", Length = 1, IsPrimaryKey = false, Comment = "データ区分: 1:ハンデ発表前(通常日曜) 2:ハンデ発表後(通常月曜) 0:該当レコード削除" },
                            new NormalFieldDefinition { Position = 12, Name = "kaisai_year", Length = 4, IsPrimaryKey = true, Comment = "開催年: 該当レース施行年 西暦4桁 yyyy形式" },
                            new NormalFieldDefinition { Position = 16, Name = "kaisai_month_day", Length = 4, IsPrimaryKey = true, Comment = "開催月日: 該当レース施行月日 各2桁 mmdd形式" },
                            new NormalFieldDefinition { Position = 20, Name = "track_code", Length = 2, IsPrimaryKey = true, Comment = "競馬場コード: 該当レース施行競馬場 <コード表 2001.競馬場コード>参照" },
                            new NormalFieldDefinition { Position = 22, Name = "kaisai_kai", Length = 2, IsPrimaryKey = true, Comment = "開催回[第N回]: 該当レース施行回 その競馬場でその年の何回目の開催かを示す" },
                            new NormalFieldDefinition { Position = 24, Name = "kaisai_day", Length = 2, IsPrimaryKey = true, Comment = "開催日目[N日目]: そのレース施行回で何日目の開催かを示す" },
                            new NormalFieldDefinition { Position = 26, Name = "race_number", Length = 2, IsPrimaryKey = true, Comment = "レース番号: 該当レース番号" },
                            new NormalFieldDefinition { Position = 28, Name = "day_of_week_code", Length = 1, IsPrimaryKey = false, Comment = "曜日コード: 該当レース施行曜日 <コード表 2002.曜日コード>参照" },
                            new NormalFieldDefinition { Position = 29, Name = "special_race_number", Length = 4, IsPrimaryKey = false, Comment = "特別競走番号: 重賞レースのみ設定 原則的には過去の同一レースと一致する番号" },
                            new NormalFieldDefinition { Position = 33, Name = "race_name_main", Length = 60, IsPrimaryKey = false, Comment = "競走名本題: 全角30文字 レース名の本題" },
                            new NormalFieldDefinition { Position = 93, Name = "race_name_sub", Length = 60, IsPrimaryKey = false, Comment = "競走名副題: 全角30文字 レース名の副題（スポンサー名や記念名など）" },
                            new NormalFieldDefinition { Position = 153, Name = "race_name_kakko", Length = 60, IsPrimaryKey = false, Comment = "競走名カッコ内: 全角30文字 レースの条件やトライアル対象レース名、レース名通称など" },
                            new NormalFieldDefinition { Position = 213, Name = "race_name_main_eng", Length = 120, IsPrimaryKey = false, Comment = "競走名本題欧字: 半角120文字" },
                            new NormalFieldDefinition { Position = 333, Name = "race_name_sub_eng", Length = 120, IsPrimaryKey = false, Comment = "競走名副題欧字: 半角120文字" },
                            new NormalFieldDefinition { Position = 453, Name = "race_name_kakko_eng", Length = 120, IsPrimaryKey = false, Comment = "競走名カッコ内欧字: 半角120文字" },
                            new NormalFieldDefinition { Position = 573, Name = "race_name_short10", Length = 20, IsPrimaryKey = false, Comment = "競走名略称10文字: 全角10文字" },
                            new NormalFieldDefinition { Position = 593, Name = "race_name_short6", Length = 12, IsPrimaryKey = false, Comment = "競走名略称6文字: 全角6文字" },
                            new NormalFieldDefinition { Position = 605, Name = "race_name_short3", Length = 6, IsPrimaryKey = false, Comment = "競走名略称3文字: 全角3文字" },
                            new NormalFieldDefinition { Position = 611, Name = "race_name_kubun", Length = 1, IsPrimaryKey = false, Comment = "競走名区分: 重賞回次を本題・副題・カッコ内のうちどれに設定すべきかを示す (0:初期値 1:本題 2:副題 3:カッコ内)" },
                            new NormalFieldDefinition { Position = 612, Name = "jyusho_kaiji", Length = 3, IsPrimaryKey = false, Comment = "重賞回次[第N回]: そのレースの重賞としての通算回数を示す" },
                            new NormalFieldDefinition { Position = 615, Name = "grade_code", Length = 1, IsPrimaryKey = false, Comment = "グレードコード: <コード表 2003.グレードコード>参照" },
                            new NormalFieldDefinition { Position = 616, Name = "race_type_code", Length = 2, IsPrimaryKey = false, Comment = "競走種別コード: <コード表 2005.競走種別コード>参照" },
                            new NormalFieldDefinition { Position = 618, Name = "race_symbol_code", Length = 3, IsPrimaryKey = false, Comment = "競走記号コード: <コード表 2006.競走記号コード>参照" },
                            new NormalFieldDefinition { Position = 621, Name = "weight_type_code", Length = 1, IsPrimaryKey = false, Comment = "重量種別コード: <コード表 2008.重量種別コード>参照" },
                            new NormalFieldDefinition { Position = 622, Name = "race_condition_2yo", Length = 3, IsPrimaryKey = false, Comment = "競走条件コード 2歳条件: <コード表 2007.競走条件コード>参照" },
                            new NormalFieldDefinition { Position = 625, Name = "race_condition_3yo", Length = 3, IsPrimaryKey = false, Comment = "競走条件コード 3歳条件: <コード表 2007.競走条件コード>参照" },
                            new NormalFieldDefinition { Position = 628, Name = "race_condition_4yo", Length = 3, IsPrimaryKey = false, Comment = "競走条件コード 4歳条件: <コード表 2007.競走条件コード>参照" },
                            new NormalFieldDefinition { Position = 631, Name = "race_condition_5yo_up", Length = 3, IsPrimaryKey = false, Comment = "競走条件コード 5歳以上条件: <コード表 2007.競走条件コード>参照" },
                            new NormalFieldDefinition { Position = 634, Name = "race_condition_youngest", Length = 3, IsPrimaryKey = false, Comment = "競走条件コード 最若年条件: 出走可能な最も馬齢が若い馬に対する条件 <コード表 2007.競走条件コード>参照" },
                            new NormalFieldDefinition { Position = 637, Name = "distance", Length = 4, IsPrimaryKey = false, Comment = "距離: 単位:メートル" },
                            new NormalFieldDefinition { Position = 641, Name = "track_code_detail", Length = 2, IsPrimaryKey = false, Comment = "トラックコード: <コード表 2009.トラックコード>参照" },
                            new NormalFieldDefinition { Position = 643, Name = "course_kubun", Length = 2, IsPrimaryKey = false, Comment = "コース区分: 半角2文字 使用するコース A～E を設定" },
                            new NormalFieldDefinition { Position = 645, Name = "handicap_announce_date", Length = 8, IsPrimaryKey = false, Comment = "ハンデ発表日: ハンデキャップレースにおいてハンデが発表された日 yyyymmdd 形式" },
                            new NormalFieldDefinition { Position = 653, Name = "registration_count", Length = 3, IsPrimaryKey = false, Comment = "登録頭数: 特別登録された頭数" },
                            new RepeatFieldDefinition
                            {
                                Position = 656,
                                RepeatCount = 300,
                                Length = 70,
                                Table = new TableDefinition {
                                    Name = "horses",
                                    Comment = "登録馬毎情報: 特別登録された各馬の詳細情報",
                                    Fields = new List<FieldDefinition>
                                    {
                                        new NormalFieldDefinition { Position = 1, Name = "sequence_number", Length = 3, IsPrimaryKey = false, Comment = "連番: 連番1～300" },
                                        new NormalFieldDefinition { Position = 4, Name = "blood_registration_number", Length = 10, IsPrimaryKey = false, Comment = "血統登録番号: 生年4桁＋品種1桁＋数字5桁" },
                                        new NormalFieldDefinition { Position = 14, Name = "horse_name", Length = 36, IsPrimaryKey = false, Comment = "馬名: 全角18文字" },
                                        new NormalFieldDefinition { Position = 50, Name = "horse_symbol_code", Length = 2, IsPrimaryKey = false, Comment = "馬記号コード: <コード表 2204.馬記号コード>参照" },
                                        new NormalFieldDefinition { Position = 52, Name = "sex_code", Length = 1, IsPrimaryKey = false, Comment = "性別コード: <コード表 2202.性別コード>参照" },
                                        new NormalFieldDefinition { Position = 53, Name = "trainer_area_code", Length = 1, IsPrimaryKey = false, Comment = "調教師東西所属コード: <コード表 2301.東西所属コード>参照" },
                                        new NormalFieldDefinition { Position = 54, Name = "trainer_code", Length = 5, IsPrimaryKey = false, Comment = "調教師コード: 調教師マスタへリンク" },
                                        new NormalFieldDefinition { Position = 59, Name = "trainer_name_short", Length = 8, IsPrimaryKey = false, Comment = "調教師名略称: 全角4文字" },
                                        new NormalFieldDefinition { Position = 67, Name = "burden_weight", Length = 3, IsPrimaryKey = false, Comment = "負担重量: 単位:0.1kg ハンデキャップレースについては月曜以降に設定" },
                                        new NormalFieldDefinition { Position = 70, Name = "exchange_kubun", Length = 1, IsPrimaryKey = false, Comment = "交流区分: 中央交流登録馬の場合に設定 0:初期値 1:地方馬 2:外国馬" }
                                    }
                                }
                            }
                        }
                    },
                    creationDateField = new FieldDefinition { Position = 4, Length = 8 }
                });

            // レース詳細レコード定義
            recordDefinitions.Add(
                new RecordDefinition
                {
                    RecordTypeId = "RA",
                    Table = new TableDefinition
                    {
                        Name = "race_detail",
                        Comment = "レース詳細レコード: 出走馬名表から成績まで、レースの詳細情報",
                        Fields = new List<FieldDefinition>
                        {
                            new NormalFieldDefinition { Position = 1, Name = "record_type_id", Length = 2, IsPrimaryKey = false, Comment = "レコード種別ID: 'RA' をセット" },
                            new NormalFieldDefinition { Position = 3, Name = "data_kubun", Length = 1, IsPrimaryKey = false, Comment = "データ区分: 1:出走馬名表(木曜) 2:出馬表(金･土曜) 3:速報成績(3着まで確定) 4:速報成績(5着まで確定) 5:速報成績(全馬着順確定) 6:速報成績(全馬着順+コーナ通過順) 7:成績(月曜) A:地方競馬 B:海外国際レース 9:レース中止 0:該当レコード削除" },
                            new NormalFieldDefinition { Position = 12, Name = "kaisai_year", Length = 4, IsPrimaryKey = true, Comment = "開催年: 該当レース施行年 西暦4桁 yyyy形式" },
                            new NormalFieldDefinition { Position = 16, Name = "kaisai_month_day", Length = 4, IsPrimaryKey = true, Comment = "開催月日: 該当レース施行月日 各2桁 mmdd形式" },
                            new NormalFieldDefinition { Position = 20, Name = "track_code", Length = 2, IsPrimaryKey = true, Comment = "競馬場コード: 該当レース施行競馬場 <コード表 2001.競馬場コード>参照" },
                            new NormalFieldDefinition { Position = 22, Name = "kaisai_kai", Length = 2, IsPrimaryKey = true, Comment = "開催回[第N回]: 該当レース施行回 その競馬場でその年の何回目の開催かを示す" },
                            new NormalFieldDefinition { Position = 24, Name = "kaisai_day", Length = 2, IsPrimaryKey = true, Comment = "開催日目[N日目]: そのレース施行日目 そのレース施行回で何日目の開催かを示す" },
                            new NormalFieldDefinition { Position = 26, Name = "race_number", Length = 2, IsPrimaryKey = true, Comment = "レース番号: 該当レース番号 海外国際レースなどでレース番号情報がない場合は任意に連番を設定" },
                            new NormalFieldDefinition { Position = 28, Name = "day_of_week_code", Length = 1, IsPrimaryKey = false, Comment = "曜日コード: 該当レース施行曜日 <コード表 2002.曜日コード>参照" },
                            new NormalFieldDefinition { Position = 29, Name = "special_race_number", Length = 4, IsPrimaryKey = false, Comment = "特別競走番号: 重賞レースのみ設定 原則的には過去の同一レースと一致する番号(多数例外有り)" },
                            new NormalFieldDefinition { Position = 33, Name = "race_name_main", Length = 60, IsPrimaryKey = false, Comment = "競走名本題: 全角30文字 レース名の本題" },
                            new NormalFieldDefinition { Position = 93, Name = "race_name_sub", Length = 60, IsPrimaryKey = false, Comment = "競走名副題: 全角30文字 レース名の副題（スポンサー名や記念名など）" },
                            new NormalFieldDefinition { Position = 153, Name = "race_name_kakko", Length = 60, IsPrimaryKey = false, Comment = "競走名カッコ内: 全角30文字 レースの条件やトライアル対象レース名、レース名通称など" },
                            new NormalFieldDefinition { Position = 213, Name = "race_name_main_eng", Length = 120, IsPrimaryKey = false, Comment = "競走名本題欧字: 半角120文字" },
                            new NormalFieldDefinition { Position = 333, Name = "race_name_sub_eng", Length = 120, IsPrimaryKey = false, Comment = "競走名副題欧字: 半角120文字" },
                            new NormalFieldDefinition { Position = 453, Name = "race_name_kakko_eng", Length = 120, IsPrimaryKey = false, Comment = "競走名カッコ内欧字: 半角120文字" },
                            new NormalFieldDefinition { Position = 573, Name = "race_name_short10", Length = 20, IsPrimaryKey = false, Comment = "競走名略称10文字: 全角10文字" },
                            new NormalFieldDefinition { Position = 593, Name = "race_name_short6", Length = 12, IsPrimaryKey = false, Comment = "競走名略称6文字: 全角6文字" },
                            new NormalFieldDefinition { Position = 605, Name = "race_name_short3", Length = 6, IsPrimaryKey = false, Comment = "競走名略称3文字: 全角3文字" },
                            new NormalFieldDefinition { Position = 611, Name = "race_name_kubun", Length = 1, IsPrimaryKey = false, Comment = "競走名区分: 重賞回次[第N回]を本題･副題･カッコ内のうちどれに設定すべきかを示す (0:初期値 1:本題 2:副題 3:カッコ内)" },
                            new NormalFieldDefinition { Position = 612, Name = "jyusho_kaiji", Length = 3, IsPrimaryKey = false, Comment = "重賞回次[第N回]: そのレースの重賞としての通算回数を示す" },
                            new NormalFieldDefinition { Position = 615, Name = "grade_code", Length = 1, IsPrimaryKey = false, Comment = "グレードコード: <コード表 2003.グレードコード>参照" },
                            new NormalFieldDefinition { Position = 616, Name = "grade_code_before", Length = 1, IsPrimaryKey = false, Comment = "変更前グレードコード: なんらかの理由により変更された場合のみ変更前の値を設定" },
                            new NormalFieldDefinition { Position = 617, Name = "race_type_code", Length = 2, IsPrimaryKey = false, Comment = "競走種別コード: <コード表 2005.競走種別コード>参照" },
                            new NormalFieldDefinition { Position = 619, Name = "race_symbol_code", Length = 3, IsPrimaryKey = false, Comment = "競走記号コード: <コード表 2006.競走記号コード>参照" },
                            new NormalFieldDefinition { Position = 622, Name = "weight_type_code", Length = 1, IsPrimaryKey = false, Comment = "重量種別コード: <コード表 2008.重量種別コード>参照" },
                            new NormalFieldDefinition { Position = 623, Name = "race_condition_2yo", Length = 3, IsPrimaryKey = false, Comment = "競走条件コード 2歳条件: 2歳馬の競走条件 <コード表 2007.競走条件コード>参照" },
                            new NormalFieldDefinition { Position = 626, Name = "race_condition_3yo", Length = 3, IsPrimaryKey = false, Comment = "競走条件コード 3歳条件: 3歳馬の競走条件 <コード表 2007.競走条件コード>参照" },
                            new NormalFieldDefinition { Position = 629, Name = "race_condition_4yo", Length = 3, IsPrimaryKey = false, Comment = "競走条件コード 4歳条件: 4歳馬の競走条件 <コード表 2007.競走条件コード>参照" },
                            new NormalFieldDefinition { Position = 632, Name = "race_condition_5yo_up", Length = 3, IsPrimaryKey = false, Comment = "競走条件コード 5歳以上条件: 5歳以上馬の競走条件 <コード表 2007.競走条件コード>参照" },
                            new NormalFieldDefinition { Position = 635, Name = "race_condition_youngest", Length = 3, IsPrimaryKey = false, Comment = "競走条件コード 最若年条件: 出走可能な最も馬齢が若い馬に対する条件 <コード表 2007.競走条件コード>参照" },
                            new NormalFieldDefinition { Position = 638, Name = "race_condition_name", Length = 60, IsPrimaryKey = false, Comment = "競走条件名称: 全角30文字 地方競馬の場合のみ設定" },
                            new NormalFieldDefinition { Position = 698, Name = "distance", Length = 4, IsPrimaryKey = false, Comment = "距離: 単位:メートル" },
                            new NormalFieldDefinition { Position = 702, Name = "distance_before", Length = 4, IsPrimaryKey = false, Comment = "変更前距離: なんらかの理由により変更された場合のみ変更前の値を設定" },
                            new NormalFieldDefinition { Position = 706, Name = "track_code_detail", Length = 2, IsPrimaryKey = false, Comment = "トラックコード: <コード表 2009.トラックコード>参照" },
                            new NormalFieldDefinition { Position = 708, Name = "track_code_detail_before", Length = 2, IsPrimaryKey = false, Comment = "変更前トラックコード: なんらかの理由により変更された場合のみ変更前の値を設定" },
                            new NormalFieldDefinition { Position = 710, Name = "course_kubun", Length = 2, IsPrimaryKey = false, Comment = "コース区分: 半角2文字 使用するコースを設定 A～E を設定 2002年以前の東京競馬場はA1、A2も存在" },
                            new NormalFieldDefinition { Position = 712, Name = "course_kubun_before", Length = 2, IsPrimaryKey = false, Comment = "変更前コース区分: なんらかの理由により変更された場合のみ変更前の値を設定" },
                            new RepeatFieldDefinition
                            {
                                Position = 714,
                                RepeatCount = 7,
                                Length = 8,
                                Table = new TableDefinition {
                                    Name = "prize",
                                    Comment = "本賞金: 単位:百円 1着～5着の本賞金 5着3同着まで考慮し繰返し7回",
                                    Fields = new List<FieldDefinition>
                                    {
                                        new NormalFieldDefinition { Position = 1, Name = "prize_money", Length = 8, IsPrimaryKey = false, Comment = "本賞金: 単位:百円" }
                                    }
                                }
                            },
                            new RepeatFieldDefinition
                            {
                                Position = 770,
                                RepeatCount = 5,
                                Length = 8,
                                Table = new TableDefinition {
                                    Name = "prize_before",
                                    Comment = "変更前本賞金: 単位:百円 同着により本賞金の分配が変更された場合のみ変更前の値を設定",
                                    Fields = new List<FieldDefinition>
                                    {
                                        new NormalFieldDefinition { Position = 1, Name = "prize_money_before", Length = 8, IsPrimaryKey = false, Comment = "変更前本賞金: 単位:百円" }
                                    }
                                }
                            },
                            new RepeatFieldDefinition
                            {
                                Position = 810,
                                RepeatCount = 5,
                                Length = 8,
                                Table = new TableDefinition
                                {
                                    Name = "additional_prize",
                                    Comment = "付加賞金: 単位:百円 1着～3着の付加賞金 3着3同着まで考慮し繰返し5回",
                                    Fields = new List<FieldDefinition>
                                    {
                                        new NormalFieldDefinition { Position = 1, Name = "additional_prize_money", Length = 8, IsPrimaryKey = false, Comment = "付加賞金: 単位:百円" }
                                    }
                                }
                            },
                            new RepeatFieldDefinition
                            {
                                Position = 850,
                                RepeatCount = 3,
                                Length = 8,
                                Table = new TableDefinition
                                {
                                    Name = "additional_prize_before",
                                    Comment = "変更前付加賞金: 単位:百円 同着により付加賞金の分配が変更された場合のみ変更前の値を設定",
                                    Fields = new List<FieldDefinition>
                                    {
                                        new NormalFieldDefinition { Position = 1, Name = "additional_prize_money_before", Length = 8, IsPrimaryKey = false, Comment = "変更前付加賞金: 単位:百円" }
                                    }
                                }
                            },
                            new NormalFieldDefinition { Position = 874, Name = "start_time", Length = 4, IsPrimaryKey = false, Comment = "発走時刻: 時分各2桁 hhmm形式" },
                            new NormalFieldDefinition { Position = 878, Name = "start_time_before", Length = 4, IsPrimaryKey = false, Comment = "変更前発走時刻: なんらかの理由により変更された場合のみ変更前の値を設定" },
                            new NormalFieldDefinition { Position = 882, Name = "registration_count", Length = 2, IsPrimaryKey = false, Comment = "登録頭数: 出走馬名表時点:出走馬名表時点での登録頭数 出馬表発表時点:出馬表発表時の登録頭数(出馬表発表前に取消した馬を除いた頭数)" },
                            new NormalFieldDefinition { Position = 884, Name = "start_count", Length = 2, IsPrimaryKey = false, Comment = "出走頭数: 実際にレースに出走した頭数 (登録頭数から出走取消と競走除外･発走除外を除いた頭数)" },
                            new NormalFieldDefinition { Position = 886, Name = "finish_count", Length = 2, IsPrimaryKey = false, Comment = "入線頭数: 出走頭数から競走中止を除いた頭数" },
                            new NormalFieldDefinition { Position = 888, Name = "weather_code", Length = 1, IsPrimaryKey = false, Comment = "天候コード: <コード表 2011.天候コード>参照" },
                            new NormalFieldDefinition { Position = 889, Name = "turf_condition_code", Length = 1, IsPrimaryKey = false, Comment = "芝馬場状態コード: <コード表 2010.馬場状態コード>参照" },
                            new NormalFieldDefinition { Position = 890, Name = "dirt_condition_code", Length = 1, IsPrimaryKey = false, Comment = "ダート馬場状態コード: <コード表 2010.馬場状態コード>参照" },
                            new RepeatFieldDefinition
                            {
                                Position = 891,
                                RepeatCount = 25,
                                Length = 3,
                                Table = new TableDefinition
                                {
                                    Name = "lap_time",
                                    Comment = "ラップタイム: 99.9秒 平地競走のみ設定 1ハロン(200メートル)毎地点での先頭馬ラップタイム",
                                    Fields = new List<FieldDefinition>
                                    {
                                        new NormalFieldDefinition { Position = 1, Name = "lap_time", Length = 3, IsPrimaryKey = false, Comment = "ラップタイム: 99.9秒 1ハロン毎の先頭馬タイム" }
                                    }
                                }
                            },
                            new NormalFieldDefinition { Position = 966, Name = "obstacle_mile_time", Length = 4, IsPrimaryKey = false, Comment = "障害マイルタイム: 障害競走のみ設定 先頭馬の1マイル(1600メートル)通過タイム" },
                            new NormalFieldDefinition { Position = 970, Name = "first_3furlong", Length = 3, IsPrimaryKey = false, Comment = "前3ハロン: 99.9秒 平地競走のみ設定 ラップタイム前半3ハロンの合計" },
                            new NormalFieldDefinition { Position = 973, Name = "first_4furlong", Length = 3, IsPrimaryKey = false, Comment = "前4ハロン: 99.9秒 平地競走のみ設定 ラップタイム前半4ハロンの合計" },
                            new NormalFieldDefinition { Position = 976, Name = "last_3furlong", Length = 3, IsPrimaryKey = false, Comment = "後3ハロン: 99.9秒 ラップタイム後半3ハロンの合計" },
                            new NormalFieldDefinition { Position = 979, Name = "last_4furlong", Length = 3, IsPrimaryKey = false, Comment = "後4ハロン: 99.9秒 ラップタイム後半4ハロンの合計" },
                            new RepeatFieldDefinition
                            {
                                Position = 982,
                                RepeatCount = 4,
                                Length = 72,
                                Table = new TableDefinition
                                {
                                    Name = "corner_passing",
                                    Comment = "コーナー通過順位: 各コーナーでの通過順位情報",
                                    Fields = new List<FieldDefinition>
                                    {
                                        new NormalFieldDefinition { Position = 1, Name = "corner", Length = 1, IsPrimaryKey = false, Comment = "コーナー: 1:1コーナー 2:2コーナー 3:3コーナー 4:4コーナー" },
                                        new NormalFieldDefinition { Position = 2, Name = "lap_count", Length = 1, IsPrimaryKey = false, Comment = "周回数: 1:1周 2:2周 3:3周" },
                                        new NormalFieldDefinition { Position = 3, Name = "passing_order", Length = 70, IsPrimaryKey = false, Comment = "各通過順位: 順位を先頭内側から設定 ():集団 =:大差 -:小差 *:先頭集団のうちで先頭の馬番" }
                                    }
                                }
                            },
                            new NormalFieldDefinition { Position = 1270, Name = "record_update_kubun", Length = 1, IsPrimaryKey = false, Comment = "レコード更新区分: 0:初期値 1:基準タイムとなったレース 2:コースレコードを更新したレース" }
                        }
                    },
                    creationDateField = new FieldDefinition { Position = 4, Length = 8 }
                });

            // 馬毎レース情報レコード定義
            recordDefinitions.Add(
                new RecordDefinition
                {
                    RecordTypeId = "SE",
                    Table = new TableDefinition
                    {
                        Name = "horse_race_info",
                        Comment = "馬毎レース情報レコード: 各馬のレース毎の詳細情報",
                        Fields = new List<FieldDefinition>
                        {
                            new NormalFieldDefinition { Position = 1, Name = "record_type_id", Length = 2, IsPrimaryKey = false, Comment = "レコード種別ID: 'SE' をセット" },
                            new NormalFieldDefinition { Position = 3, Name = "data_kubun", Length = 1, IsPrimaryKey = false, Comment = "データ区分: 1:出走馬名表(木曜) 2:出馬表(金･土曜) 3:速報成績(3着まで確定) 4:速報成績(5着まで確定) 5:速報成績(全馬着順確定) 6:速報成績(全馬着順+コーナ通過順) 7:成績(月曜) A:地方競馬 B:海外国際レース 9:レース中止 0:該当レコード削除" },
                            new NormalFieldDefinition { Position = 12, Name = "kaisai_year", Length = 4, IsPrimaryKey = true, Comment = "開催年: 該当レース施行年 西暦4桁 yyyy形式" },
                            new NormalFieldDefinition { Position = 16, Name = "kaisai_month_day", Length = 4, IsPrimaryKey = true, Comment = "開催月日: 該当レース施行月日 各2桁 mmdd形式" },
                            new NormalFieldDefinition { Position = 20, Name = "track_code", Length = 2, IsPrimaryKey = true, Comment = "競馬場コード: 該当レース施行競馬場 <コード表 2001.競馬場コード>参照" },
                            new NormalFieldDefinition { Position = 22, Name = "kaisai_kai", Length = 2, IsPrimaryKey = true, Comment = "開催回[第N回]: 該当レース施行回 その競馬場でその年の何回目の開催かを示す" },
                            new NormalFieldDefinition { Position = 24, Name = "kaisai_day", Length = 2, IsPrimaryKey = true, Comment = "開催日目[N日目]: そのレース施行回で何日目の開催かを示す" },
                            new NormalFieldDefinition { Position = 26, Name = "race_number", Length = 2, IsPrimaryKey = true, Comment = "レース番号: 該当レース番号" },
                            new NormalFieldDefinition { Position = 28, Name = "frame_number", Length = 1, IsPrimaryKey = false, Comment = "枠番" },
                            new NormalFieldDefinition { Position = 29, Name = "horse_number", Length = 2, IsPrimaryKey = true, Comment = "馬番: 特定のレース及び海外レースについては、特記事項を参照" },
                            new NormalFieldDefinition { Position = 31, Name = "blood_registration_number", Length = 10, IsPrimaryKey = false, Comment = "血統登録番号: 生年(西暦)4桁＋品種1桁<コード表2201.品種コード>参照＋数字5桁" },
                            new NormalFieldDefinition { Position = 41, Name = "horse_name", Length = 36, IsPrimaryKey = false, Comment = "馬名: 通常全角18文字。海外レースにおける外国馬の場合のみ全角と半角が混在" },
                            new NormalFieldDefinition { Position = 77, Name = "horse_symbol_code", Length = 2, IsPrimaryKey = false, Comment = "馬記号コード: <コード表 2204.馬記号コード>参照" },
                            new NormalFieldDefinition { Position = 79, Name = "sex_code", Length = 1, IsPrimaryKey = false, Comment = "性別コード: <コード表 2202.性別コード>参照" },
                            new NormalFieldDefinition { Position = 80, Name = "breed_code", Length = 1, IsPrimaryKey = false, Comment = "品種コード: <コード表 2201.品種コード>参照" },
                            new NormalFieldDefinition { Position = 81, Name = "coat_color_code", Length = 2, IsPrimaryKey = false, Comment = "毛色コード: <コード表 2203.毛色コード>参照" },
                            new NormalFieldDefinition { Position = 83, Name = "horse_age", Length = 2, IsPrimaryKey = false, Comment = "馬齢: 出走当時の馬齢 (注)2000年以前は数え年表記 2001年以降は満年齢表記" },
                            new NormalFieldDefinition { Position = 85, Name = "trainer_area_code", Length = 1, IsPrimaryKey = false, Comment = "東西所属コード: <コード表 2301.東西所属コード>参照" },
                            new NormalFieldDefinition { Position = 86, Name = "trainer_code", Length = 5, IsPrimaryKey = false, Comment = "調教師コード: 調教師マスタへリンク" },
                            new NormalFieldDefinition { Position = 91, Name = "trainer_name_short", Length = 8, IsPrimaryKey = false, Comment = "調教師名略称: 全角4文字" },
                            new NormalFieldDefinition { Position = 99, Name = "owner_code", Length = 6, IsPrimaryKey = false, Comment = "馬主コード: 馬主マスタへリンク" },
                            new NormalFieldDefinition { Position = 105, Name = "owner_name_no_corp", Length = 64, IsPrimaryKey = false, Comment = "馬主名(法人格無): 全角32文字 ～ 半角64文字 (全角と半角が混在) 株式会社、有限会社などの法人格を示す文字列が頭もしくは末尾にある場合にそれを削除したものを設定。また、外国馬主の場合は、馬主マスタの8.馬主名欧字の頭64バイトを設定" },
                            new NormalFieldDefinition { Position = 169, Name = "racing_color", Length = 60, IsPrimaryKey = false, Comment = "服色標示: 全角30文字 馬主毎に指定される騎手の勝負服の色・模様を示す (レーシングプログラムに記載されているもの) (例)\"水色，赤山形一本輪，水色袖\"" },
                            new NormalFieldDefinition { Position = 229, Name = "reserve1", Length = 60, IsPrimaryKey = false, Comment = "予備" },
                            new NormalFieldDefinition { Position = 289, Name = "burden_weight", Length = 3, IsPrimaryKey = false, Comment = "負担重量: 単位0.1kg" },
                            new NormalFieldDefinition { Position = 292, Name = "burden_weight_before", Length = 3, IsPrimaryKey = false, Comment = "変更前負担重量: なんらかの理由により変更された場合のみ変更前の値を設定" },
                            new NormalFieldDefinition { Position = 295, Name = "blinker_use_kubun", Length = 1, IsPrimaryKey = false, Comment = "ブリンカー使用区分: 0:未使用 1:使用" },
                            new NormalFieldDefinition { Position = 296, Name = "reserve2", Length = 1, IsPrimaryKey = false, Comment = "予備" },
                            new NormalFieldDefinition { Position = 297, Name = "jockey_code", Length = 5, IsPrimaryKey = false, Comment = "騎手コード: 騎手マスタへリンク" },
                            new NormalFieldDefinition { Position = 302, Name = "jockey_code_before", Length = 5, IsPrimaryKey = false, Comment = "変更前騎手コード: なんらかの理由により変更された場合のみ変更前の値を設定" },
                            new NormalFieldDefinition { Position = 307, Name = "jockey_name_short", Length = 8, IsPrimaryKey = false, Comment = "騎手名略称: 全角4文字" },
                            new NormalFieldDefinition { Position = 315, Name = "jockey_name_short_before", Length = 8, IsPrimaryKey = false, Comment = "変更前騎手名略称: なんらかの理由により変更された場合のみ変更前の値を設定" },
                            new NormalFieldDefinition { Position = 323, Name = "jockey_apprentice_code", Length = 1, IsPrimaryKey = false, Comment = "騎手見習コード: <コード表 2303.騎手見習コード>参照" },
                            new NormalFieldDefinition { Position = 324, Name = "jockey_apprentice_code_before", Length = 1, IsPrimaryKey = false, Comment = "変更前騎手見習コード: なんらかの理由により変更された場合のみ変更前の値を設定" },
                            new NormalFieldDefinition { Position = 325, Name = "horse_weight", Length = 3, IsPrimaryKey = false, Comment = "馬体重: 単位:kg 002Kg～998Kgまでが有効値 999:今走計量不能 000:出走取消" },
                            new NormalFieldDefinition { Position = 328, Name = "weight_change_sign", Length = 1, IsPrimaryKey = false, Comment = "増減符号: +:増加 -:減少 スペース:その他" },
                            new NormalFieldDefinition { Position = 329, Name = "weight_change", Length = 3, IsPrimaryKey = false, Comment = "増減差: 単位:kg 001Kg～998Kgまでが有効値 999:計量不能 000:前差なし スペース:初出走、ただし出走取消の場合もスペースを設定。地方馬については初出走かつ計量不能の場合でも\"999\"を設定。" },
                            new NormalFieldDefinition { Position = 332, Name = "abnormal_kubun_code", Length = 1, IsPrimaryKey = false, Comment = "異常区分コード: <コード表 2101.異常区分コード>参照" },
                            new NormalFieldDefinition { Position = 333, Name = "entry_order", Length = 2, IsPrimaryKey = false, Comment = "入線順位: 失格、降着確定前の順位" },
                            new NormalFieldDefinition { Position = 335, Name = "final_order", Length = 2, IsPrimaryKey = false, Comment = "確定着順: 失格、降着時は入線順位と異なる" },
                            new NormalFieldDefinition { Position = 337, Name = "dead_heat_kubun", Length = 1, IsPrimaryKey = false, Comment = "同着区分: 0:同着馬なし 1:同着馬あり" },
                            new NormalFieldDefinition { Position = 338, Name = "dead_heat_count", Length = 1, IsPrimaryKey = false, Comment = "同着頭数: 0:初期値 1:自身以外に同着1頭 2:自身以外に同着2頭" },
                            new NormalFieldDefinition { Position = 339, Name = "finish_time", Length = 4, IsPrimaryKey = false, Comment = "走破タイム: 9分99秒9で設定" },
                            new NormalFieldDefinition { Position = 343, Name = "time_diff_code", Length = 3, IsPrimaryKey = false, Comment = "着差コード: 前馬との着差 <コード表 2102.着差コード>参照" },
                            new NormalFieldDefinition { Position = 346, Name = "time_diff_code_plus", Length = 3, IsPrimaryKey = false, Comment = "＋着差コード: 前馬が失格、降着発生時に設定 前馬と前馬の前馬との着差" },
                            new NormalFieldDefinition { Position = 349, Name = "time_diff_code_plus2", Length = 3, IsPrimaryKey = false, Comment = "＋＋着差コード: 前馬2頭が失格、降着発生時に設定" },
                            new NormalFieldDefinition { Position = 352, Name = "corner1_order", Length = 2, IsPrimaryKey = false, Comment = "1コーナーでの順位" },
                            new NormalFieldDefinition { Position = 354, Name = "corner2_order", Length = 2, IsPrimaryKey = false, Comment = "2コーナーでの順位" },
                            new NormalFieldDefinition { Position = 356, Name = "corner3_order", Length = 2, IsPrimaryKey = false, Comment = "3コーナーでの順位" },
                            new NormalFieldDefinition { Position = 358, Name = "corner4_order", Length = 2, IsPrimaryKey = false, Comment = "4コーナーでの順位" },
                            new NormalFieldDefinition { Position = 360, Name = "win_odds", Length = 4, IsPrimaryKey = false, Comment = "単勝オッズ: 999.9倍で設定 出走取消し等は初期値を設定" },
                            new NormalFieldDefinition { Position = 364, Name = "win_popularity", Length = 2, IsPrimaryKey = false, Comment = "単勝人気順: 出走取消し等は初期値を設定" },
                            new NormalFieldDefinition { Position = 366, Name = "main_prize_money", Length = 8, IsPrimaryKey = false, Comment = "獲得本賞金: 単位:百円 該当レースで獲得した本賞金" },
                            new NormalFieldDefinition { Position = 374, Name = "additional_prize_money", Length = 8, IsPrimaryKey = false, Comment = "獲得付加賞金: 単位:百円 該当レースで獲得した付加賞金" },
                            new NormalFieldDefinition { Position = 382, Name = "reserve3", Length = 3, IsPrimaryKey = false, Comment = "予備" },
                            new NormalFieldDefinition { Position = 385, Name = "reserve4", Length = 3, IsPrimaryKey = false, Comment = "予備" },
                            new NormalFieldDefinition { Position = 388, Name = "last_4furlong_time", Length = 3, IsPrimaryKey = false, Comment = "後4ハロンタイム: 単位:99.9秒 出走取消･競走除外･発走除外･競走中止･タイムオーバーの場合は\"999\"を設定 基本的には後3ハロンのみ設定(後4ハロンは初期値) ただし、過去分のデータは後4ハロンが設定されているものもある(その場合は後3ハロンが初期値) 障害レースの場合は後3ハロンに該当馬の当該レースでの1F平均タイムを設定(後4ハロンは初期値)" },
                            new NormalFieldDefinition { Position = 391, Name = "last_3furlong_time", Length = 3, IsPrimaryKey = false, Comment = "後3ハロンタイム" },
                            new RepeatFieldDefinition
                            {
                                Position = 394,
                                RepeatCount = 3,
                                Length = 46,
                                Table = new TableDefinition
                                {
                                    Name = "rival",
                                    Comment = "1着馬(相手馬)情報: 同着を考慮して繰返し3回 自身が1着の場合は2着馬を設定",
                                    Fields = new List<FieldDefinition>
                                    {
                                        new NormalFieldDefinition { Position = 1, Name = "rival_blood_registration_number", Length = 10, IsPrimaryKey = false, Comment = "血統登録番号: 生年(西暦)4桁＋品種1桁<コード表2201.品種コード>参照＋数字5桁" },
                                        new NormalFieldDefinition { Position = 11, Name = "rival_horse_name", Length = 36, IsPrimaryKey = false, Comment = "馬名: 通常全角18文字。海外レースにおける外国馬の場合のみ全角と半角が混在。" }
                                    }
                                }
                            },
                            new NormalFieldDefinition { Position = 532, Name = "time_difference", Length = 4, IsPrimaryKey = false, Comment = "タイム差: 1着馬とのタイム差を設定(自身が1着の場合は2着馬を設定) 符号(+または-)+99秒9 符号は1着:-, 2着以下:+ 出走取消･競走除外･発走除外･競走中止の場合は\"9999\"を設定" },
                            new NormalFieldDefinition { Position = 536, Name = "record_update_kubun", Length = 1, IsPrimaryKey = false, Comment = "レコード更新区分: 0:初期値 1:基準タイムとなったレース 2:コースレコードを更新したレース" },
                            new NormalFieldDefinition { Position = 537, Name = "mining_kubun", Length = 1, IsPrimaryKey = false, Comment = "マイニング区分: 1:前日 2:当日 3:直前 ただし、確定成績登録時に3:直前のみ設定" },
                            new NormalFieldDefinition { Position = 538, Name = "mining_predicted_time", Length = 5, IsPrimaryKey = false, Comment = "マイニング予想走破タイム: 9分99秒99で設定" },
                            new NormalFieldDefinition { Position = 543, Name = "mining_error_plus", Length = 4, IsPrimaryKey = false, Comment = "マイニング予想誤差(信頼度)＋: 99秒99で設定 予想タイムの＋誤差を設定(＋方向の誤差。予想走破タイムに対して早くなる方向。予想走破タイムからマイナスする。)" },
                            new NormalFieldDefinition { Position = 547, Name = "mining_error_minus", Length = 4, IsPrimaryKey = false, Comment = "マイニング予想誤差(信頼度)－: 99秒99で設定 予想タイムの－誤差を設定(－方向の誤差。予想走破タイムに対して遅くなる方向。予想走破タイムにプラスする。)" },
                            new NormalFieldDefinition { Position = 551, Name = "mining_predicted_order", Length = 2, IsPrimaryKey = false, Comment = "マイニング予想順位: 01～18位を設定" },
                            new NormalFieldDefinition { Position = 553, Name = "running_style_judgment", Length = 1, IsPrimaryKey = false, Comment = "今回レース脚質判定: 1:逃 2:先 3:差 4:追 0:初期値" }
                        }
                    },
                    creationDateField = new FieldDefinition { Position = 4, Length = 8 }
                });

            // 払戻レコード定義
            recordDefinitions.Add(
                new RecordDefinition
                {
                    RecordTypeId = "HR",
                    Table = new TableDefinition
                    {
                        Name = "payoff",
                        Comment = "払戻レコード: レースの払戻金情報",
                        Fields = new List<FieldDefinition>
                        {
                            new NormalFieldDefinition { Position = 1, Name = "record_type_id", Length = 2, IsPrimaryKey = false, Comment = "レコード種別ID: 'HR' をセット" },
                            new NormalFieldDefinition { Position = 3, Name = "data_kubun", Length = 1, IsPrimaryKey = false, Comment = "データ区分: 1:速報成績(払戻金確定) 2:成績(月曜) 9:レース中止 0:該当レコード削除" },
                            new NormalFieldDefinition { Position = 12, Name = "kaisai_year", Length = 4, IsPrimaryKey = true, Comment = "開催年: 該当レース施行年 西暦4桁 yyyy形式" },
                            new NormalFieldDefinition { Position = 16, Name = "kaisai_month_day", Length = 4, IsPrimaryKey = true, Comment = "開催月日: 該当レース施行月日 各2桁 mmdd形式" },
                            new NormalFieldDefinition { Position = 20, Name = "track_code", Length = 2, IsPrimaryKey = true, Comment = "競馬場コード: 該当レース施行競馬場 <コード表 2001.競馬場コード>参照" },
                            new NormalFieldDefinition { Position = 22, Name = "kaisai_kai", Length = 2, IsPrimaryKey = true, Comment = "開催回[第N回]: 該当レース施行回 その競馬場でその年の何回目の開催かを示す" },
                            new NormalFieldDefinition { Position = 24, Name = "kaisai_day", Length = 2, IsPrimaryKey = true, Comment = "開催日目[N日目]: そのレース施行回で何日目の開催かを示す" },
                            new NormalFieldDefinition { Position = 26, Name = "race_number", Length = 2, IsPrimaryKey = true, Comment = "レース番号: 該当レース番号" },
                            new NormalFieldDefinition { Position = 28, Name = "registration_count", Length = 2, IsPrimaryKey = false, Comment = "登録頭数: 出馬表発表時の登録頭数" },
                            new NormalFieldDefinition { Position = 30, Name = "start_count", Length = 2, IsPrimaryKey = false, Comment = "出走頭数: 登録頭数から出走取消と競走除外･発走除外を除いた頭数" },
                            new NormalFieldDefinition { Position = 32, Name = "failed_flag_tansho", Length = 1, IsPrimaryKey = false, Comment = "不成立フラグ 単勝: 0:不成立なし 1:不成立あり" },
                            new NormalFieldDefinition { Position = 33, Name = "failed_flag_fukusho", Length = 1, IsPrimaryKey = false, Comment = "不成立フラグ 複勝: 0:不成立なし 1:不成立あり" },
                            new NormalFieldDefinition { Position = 34, Name = "failed_flag_wakuren", Length = 1, IsPrimaryKey = false, Comment = "不成立フラグ 枠連: 0:不成立なし 1:不成立あり" },
                            new NormalFieldDefinition { Position = 35, Name = "failed_flag_umaren", Length = 1, IsPrimaryKey = false, Comment = "不成立フラグ 馬連: 0:不成立なし 1:不成立あり" },
                            new NormalFieldDefinition { Position = 36, Name = "failed_flag_wide", Length = 1, IsPrimaryKey = false, Comment = "不成立フラグ ワイド: 0:不成立なし 1:不成立あり" },
                            new NormalFieldDefinition { Position = 37, Name = "reserve1", Length = 1, IsPrimaryKey = false, Comment = "予備" },
                            new NormalFieldDefinition { Position = 38, Name = "failed_flag_umatan", Length = 1, IsPrimaryKey = false, Comment = "不成立フラグ 馬単: 0:不成立なし 1:不成立あり" },
                            new NormalFieldDefinition { Position = 39, Name = "failed_flag_sanrenpuku", Length = 1, IsPrimaryKey = false, Comment = "不成立フラグ 3連複: 0:不成立なし 1:不成立あり" },
                            new NormalFieldDefinition { Position = 40, Name = "failed_flag_sanrentan", Length = 1, IsPrimaryKey = false, Comment = "不成立フラグ 3連単: 0:不成立なし 1:不成立あり" },
                            new NormalFieldDefinition { Position = 41, Name = "special_flag_tansho", Length = 1, IsPrimaryKey = false, Comment = "特払フラグ 単勝: 0:特払なし 1:特払あり" },
                            new NormalFieldDefinition { Position = 42, Name = "special_flag_fukusho", Length = 1, IsPrimaryKey = false, Comment = "特払フラグ 複勝: 0:特払なし 1:特払あり" },
                            new NormalFieldDefinition { Position = 43, Name = "special_flag_wakuren", Length = 1, IsPrimaryKey = false, Comment = "特払フラグ 枠連: 0:特払なし 1:特払あり" },
                            new NormalFieldDefinition { Position = 44, Name = "special_flag_umaren", Length = 1, IsPrimaryKey = false, Comment = "特払フラグ 馬連: 0:特払なし 1:特払あり" },
                            new NormalFieldDefinition { Position = 45, Name = "special_flag_wide", Length = 1, IsPrimaryKey = false, Comment = "特払フラグ ワイド: 0:特払なし 1:特払あり" },
                            new NormalFieldDefinition { Position = 46, Name = "reserve2", Length = 1, IsPrimaryKey = false, Comment = "予備" },
                            new NormalFieldDefinition { Position = 47, Name = "special_flag_umatan", Length = 1, IsPrimaryKey = false, Comment = "特払フラグ 馬単: 0:特払なし 1:特払あり" },
                            new NormalFieldDefinition { Position = 48, Name = "special_flag_sanrenpuku", Length = 1, IsPrimaryKey = false, Comment = "特払フラグ 3連複: 0:特払なし 1:特払あり" },
                            new NormalFieldDefinition { Position = 49, Name = "special_flag_sanrentan", Length = 1, IsPrimaryKey = false, Comment = "特払フラグ 3連単: 0:特払なし 1:特払あり" },
                            new NormalFieldDefinition { Position = 50, Name = "refund_flag_tansho", Length = 1, IsPrimaryKey = false, Comment = "返還フラグ 単勝: 0:返還なし 1:返還あり" },
                            new NormalFieldDefinition { Position = 51, Name = "refund_flag_fukusho", Length = 1, IsPrimaryKey = false, Comment = "返還フラグ 複勝: 0:返還なし 1:返還あり" },
                            new NormalFieldDefinition { Position = 52, Name = "refund_flag_wakuren", Length = 1, IsPrimaryKey = false, Comment = "返還フラグ 枠連: 0:返還なし 1:返還あり" },
                            new NormalFieldDefinition { Position = 53, Name = "refund_flag_umaren", Length = 1, IsPrimaryKey = false, Comment = "返還フラグ 馬連: 0:返還なし 1:返還あり" },
                            new NormalFieldDefinition { Position = 54, Name = "refund_flag_wide", Length = 1, IsPrimaryKey = false, Comment = "返還フラグ ワイド: 0:返還なし 1:返還あり" },
                            new NormalFieldDefinition { Position = 55, Name = "reserve3", Length = 1, IsPrimaryKey = false, Comment = "予備" },
                            new NormalFieldDefinition { Position = 56, Name = "refund_flag_umatan", Length = 1, IsPrimaryKey = false, Comment = "返還フラグ 馬単: 0:返還なし 1:返還あり" },
                            new NormalFieldDefinition { Position = 57, Name = "refund_flag_sanrenpuku", Length = 1, IsPrimaryKey = false, Comment = "返還フラグ 3連複: 0:返還なし 1:返還あり" },
                            new NormalFieldDefinition { Position = 58, Name = "refund_flag_sanrentan", Length = 1, IsPrimaryKey = false, Comment = "返還フラグ 3連単: 0:返還なし 1:返還あり" },
                            new NormalFieldDefinition { Position = 59, Name = "refund_horse_info", Length = 28, IsPrimaryKey = false, Comment = "返還馬番情報(馬番01～28): 0:返還なし 1:返還あり 発売後取消しとなり返還対象となった馬番のエリアに\"1\"を設定" },
                            new NormalFieldDefinition { Position = 87, Name = "refund_frame_info", Length = 8, IsPrimaryKey = false, Comment = "返還枠番情報(枠番1～8): 0:返還なし 1:返還あり 発売後取消しとなり返還対象となった枠番のエリアに\"1\"を設定" },
                            new NormalFieldDefinition { Position = 95, Name = "refund_same_frame_info", Length = 8, IsPrimaryKey = false, Comment = "返還同枠情報(枠番1～8): 0:返還なし 1:返還あり 発売後取消しとなり同枠のみ取消しとなった枠番のエリアに\"1\"を設定" },
                            new RepeatFieldDefinition
                            {
                                Position = 103,
                                RepeatCount = 3,
                                Length = 13,
                                Table = new TableDefinition
                                {
                                    Name = "tansho_payoff",
                                    Comment = "単勝払戻: 3同着まで考慮し繰返し3回",
                                    Fields = new List<FieldDefinition>
                                    {
                                        new NormalFieldDefinition { Position = 1, Name = "horse_number", Length = 2, IsPrimaryKey = false, Comment = "単勝的中馬番: 00:発売なし、特払、不成立" },
                                        new NormalFieldDefinition { Position = 3, Name = "payoff_money", Length = 9, IsPrimaryKey = false, Comment = "単勝払戻金: 特払、不成立の金額が入る" },
                                        new NormalFieldDefinition { Position = 12, Name = "popularity", Length = 2, IsPrimaryKey = false, Comment = "単勝人気順" }
                                    }
                                }
                            },
                            new RepeatFieldDefinition
                            {
                                Position = 142,
                                RepeatCount = 5,
                                Length = 13,
                                Table = new TableDefinition
                                {
                                    Name = "fukusho_payoff",
                                    Comment = "複勝払戻: 3同着まで考慮し繰返し5回",
                                    Fields = new List<FieldDefinition>
                                    {
                                        new NormalFieldDefinition { Position = 1, Name = "horse_number", Length = 2, IsPrimaryKey = false, Comment = "複勝的中馬番: 00:発売なし、特払、不成立" },
                                        new NormalFieldDefinition { Position = 3, Name = "payoff_money", Length = 9, IsPrimaryKey = false, Comment = "複勝払戻金: 特払、不成立の金額が入る" },
                                        new NormalFieldDefinition { Position = 12, Name = "popularity", Length = 2, IsPrimaryKey = false, Comment = "複勝人気順" }
                                    }
                                }
                            },
                            new RepeatFieldDefinition
                            {
                                Position = 207,
                                RepeatCount = 3,
                                Length = 13,
                                Table = new TableDefinition
                                {
                                    Name = "wakuren_payoff",
                                    Comment = "枠連払戻: 3同着まで考慮し繰返し3回",
                                    Fields = new List<FieldDefinition>
                                    {
                                        new NormalFieldDefinition { Position = 1, Name = "combination", Length = 2, IsPrimaryKey = false, Comment = "枠連的中組番: 00:発売なし、特払、不成立" },
                                        new NormalFieldDefinition { Position = 3, Name = "payoff_money", Length = 9, IsPrimaryKey = false, Comment = "枠連払戻金: 特払、不成立の金額が入る" },
                                        new NormalFieldDefinition { Position = 12, Name = "popularity", Length = 2, IsPrimaryKey = false, Comment = "枠連人気順" }
                                    }
                                }
                            },
                            new RepeatFieldDefinition
                            {
                                Position = 246,
                                RepeatCount = 3,
                                Length = 16,
                                Table = new TableDefinition
                                {
                                    Name = "umaren_payoff",
                                    Comment = "馬連払戻: 3同着まで考慮し繰返し3回",
                                    Fields = new List<FieldDefinition>
                                    {
                                        new NormalFieldDefinition { Position = 1, Name = "combination", Length = 4, IsPrimaryKey = false, Comment = "馬連的中馬番組合: 0000:発売なし、特払、不成立" },
                                        new NormalFieldDefinition { Position = 5, Name = "payoff_money", Length = 9, IsPrimaryKey = false, Comment = "馬連払戻金: 特払、不成立の金額が入る" },
                                        new NormalFieldDefinition { Position = 14, Name = "popularity", Length = 3, IsPrimaryKey = false, Comment = "馬連人気順" }
                                    }
                                }
                            },
                            new RepeatFieldDefinition
                            {
                                Position = 294,
                                RepeatCount = 7,
                                Length = 16,
                                Table = new TableDefinition
                                {
                                    Name = "wide_payoff",
                                    Comment = "ワイド払戻: 3同着まで考慮し繰返し7回",
                                    Fields = new List<FieldDefinition>
                                    {
                                        new NormalFieldDefinition { Position = 1, Name = "combination", Length = 4, IsPrimaryKey = false, Comment = "ワイド的中馬番組合: 0000:発売なし、特払、不成立" },
                                        new NormalFieldDefinition { Position = 5, Name = "payoff_money", Length = 9, IsPrimaryKey = false, Comment = "ワイド払戻金: 特払、不成立の金額が入る" },
                                        new NormalFieldDefinition { Position = 14, Name = "popularity", Length = 3, IsPrimaryKey = false, Comment = "ワイド人気順" }
                                    }
                                }
                            },
                            new RepeatFieldDefinition
                            {
                                Position = 406,
                                RepeatCount = 3,
                                Length = 16,
                                Table = new TableDefinition
                                {
                                    Name = "reserve_payoff",
                                    Comment = "予備払戻情報",
                                    Fields = new List<FieldDefinition>
                                    {
                                        new NormalFieldDefinition { Position = 1, Name = "reserve_combination", Length = 4, IsPrimaryKey = false, Comment = "予備" },
                                        new NormalFieldDefinition { Position = 5, Name = "reserve_payoff_money", Length = 9, IsPrimaryKey = false, Comment = "予備" },
                                        new NormalFieldDefinition { Position = 14, Name = "reserve_popularity", Length = 3, IsPrimaryKey = false, Comment = "予備" }
                                    }
                                }
                            },
                            new RepeatFieldDefinition
                            {
                                Position = 454,
                                RepeatCount = 6,
                                Length = 16,
                                Table = new TableDefinition
                                {
                                    Name = "umatan_payoff",
                                    Comment = "馬単払戻: 3同着まで考慮し繰返し6回",
                                    Fields = new List<FieldDefinition>
                                    {
                                        new NormalFieldDefinition { Position = 1, Name = "combination", Length = 4, IsPrimaryKey = false, Comment = "馬単的中馬番組合: 0000:発売なし、特払、不成立" },
                                        new NormalFieldDefinition { Position = 5, Name = "payoff_money", Length = 9, IsPrimaryKey = false, Comment = "馬単払戻金: 特払、不成立の金額が入る" },
                                        new NormalFieldDefinition { Position = 14, Name = "popularity", Length = 3, IsPrimaryKey = false, Comment = "馬単人気順" }
                                    }
                                }
                            },
                            new RepeatFieldDefinition
                            {
                                Position = 550,
                                RepeatCount = 3,
                                Length = 18,
                                Table = new TableDefinition
                                {
                                    Name = "sanrenpuku_payoff",
                                    Comment = "3連複払戻: 3同着まで考慮し繰返し3回",
                                    Fields = new List<FieldDefinition>
                                    {
                                        new NormalFieldDefinition { Position = 1, Name = "combination", Length = 6, IsPrimaryKey = false, Comment = "3連複的中馬番組合: 000000:発売なし、特払、不成立" },
                                        new NormalFieldDefinition { Position = 7, Name = "payoff_money", Length = 9, IsPrimaryKey = false, Comment = "3連複払戻金: 特払、不成立の金額が入る" },
                                        new NormalFieldDefinition { Position = 16, Name = "popularity", Length = 3, IsPrimaryKey = false, Comment = "3連複人気順" }
                                    }
                                }
                            },
                            new RepeatFieldDefinition
                            {
                                Position = 604,
                                RepeatCount = 6,
                                Length = 19,
                                Table = new TableDefinition
                                {
                                    Name = "sanrentan_payoff",
                                    Comment = "3連単払戻: 3同着まで考慮し繰返し6回",
                                    Fields = new List<FieldDefinition>
                                    {
                                        new NormalFieldDefinition { Position = 1, Name = "combination", Length = 6, IsPrimaryKey = false, Comment = "3連単的中馬番組合: 000000:発売なし、特払、不成立" },
                                        new NormalFieldDefinition { Position = 7, Name = "payoff_money", Length = 9, IsPrimaryKey = false, Comment = "3連単払戻金: 特払、不成立の金額が入る" },
                                        new NormalFieldDefinition { Position = 16, Name = "popularity", Length = 4, IsPrimaryKey = false, Comment = "3連単人気順" }
                                    }
                                }
                            }
                        }
                    },
                    creationDateField = new FieldDefinition { Position = 4, Length = 8 }
                });

            // 票数１レコード定義
            recordDefinitions.Add(
                new RecordDefinition
                {
                    RecordTypeId = "H1",
                    Table = new TableDefinition
                    {
                        Name = "vote_count_1",
                        Comment = "票数１レコード: 各賭式の票数情報（前日売最終～確定）",
                        Fields = new List<FieldDefinition>
                        {
                            new NormalFieldDefinition { Position = 1, Name = "record_type_id", Length = 2, IsPrimaryKey = false, Comment = "レコード種別ID: 'H1' をセット" },
                            new NormalFieldDefinition { Position = 3, Name = "data_kubun", Length = 1, IsPrimaryKey = false, Comment = "データ区分: 2:前日売最終 4:確定 5:確定(月曜) 9:レース中止 0:該当レコード削除" },
                            new NormalFieldDefinition { Position = 12, Name = "kaisai_year", Length = 4, IsPrimaryKey = true, Comment = "開催年: 該当レース施行年 西暦4桁 yyyy形式" },
                            new NormalFieldDefinition { Position = 16, Name = "kaisai_month_day", Length = 4, IsPrimaryKey = true, Comment = "開催月日: 該当レース施行月日 各2桁 mmdd形式" },
                            new NormalFieldDefinition { Position = 20, Name = "track_code", Length = 2, IsPrimaryKey = true, Comment = "競馬場コード: 該当レース施行競馬場 <コード表 2001.競馬場コード>参照" },
                            new NormalFieldDefinition { Position = 22, Name = "kaisai_kai", Length = 2, IsPrimaryKey = true, Comment = "開催回[第N回]: 該当レース施行回 その競馬場でその年の何回目の開催かを示す" },
                            new NormalFieldDefinition { Position = 24, Name = "kaisai_day", Length = 2, IsPrimaryKey = true, Comment = "開催日目[N日目]: そのレース施行回で何日目の開催かを示す" },
                            new NormalFieldDefinition { Position = 26, Name = "race_number", Length = 2, IsPrimaryKey = true, Comment = "レース番号: 該当レース番号" },
                            new NormalFieldDefinition { Position = 28, Name = "registration_count", Length = 2, IsPrimaryKey = false, Comment = "登録頭数: 出馬表発表時の登録頭数" },
                            new NormalFieldDefinition { Position = 30, Name = "start_count", Length = 2, IsPrimaryKey = false, Comment = "出走頭数: 登録頭数から出走取消と競走除外･発走除外を除いた頭数" },
                            new NormalFieldDefinition { Position = 32, Name = "sale_flag_tansho", Length = 1, IsPrimaryKey = false, Comment = "発売フラグ 単勝: 0:発売なし 1:発売前取消 3:発売後取消 7:発売あり" },
                            new NormalFieldDefinition { Position = 33, Name = "sale_flag_fukusho", Length = 1, IsPrimaryKey = false, Comment = "発売フラグ 複勝: 0:発売なし 1:発売前取消 3:発売後取消 7:発売あり" },
                            new NormalFieldDefinition { Position = 34, Name = "sale_flag_wakuren", Length = 1, IsPrimaryKey = false, Comment = "発売フラグ 枠連: 0:発売なし 1:発売前取消 3:発売後取消 7:発売あり" },
                            new NormalFieldDefinition { Position = 35, Name = "sale_flag_umaren", Length = 1, IsPrimaryKey = false, Comment = "発売フラグ 馬連: 0:発売なし 1:発売前取消 3:発売後取消 7:発売あり" },
                            new NormalFieldDefinition { Position = 36, Name = "sale_flag_wide", Length = 1, IsPrimaryKey = false, Comment = "発売フラグ ワイド: 0:発売なし 1:発売前取消 3:発売後取消 7:発売あり" },
                            new NormalFieldDefinition { Position = 37, Name = "sale_flag_umatan", Length = 1, IsPrimaryKey = false, Comment = "発売フラグ 馬単: 0:発売なし 1:発売前取消 3:発売後取消 7:発売あり" },
                            new NormalFieldDefinition { Position = 38, Name = "sale_flag_sanrenpuku", Length = 1, IsPrimaryKey = false, Comment = "発売フラグ 3連複: 0:発売なし 1:発売前取消 3:発売後取消 7:発売あり" },
                            new NormalFieldDefinition { Position = 39, Name = "fukusho_chakubarai_key", Length = 1, IsPrimaryKey = false, Comment = "複勝着払キー: 0:複勝発売なし 2:2着まで払い 3:3着まで払い" },
                            new NormalFieldDefinition { Position = 40, Name = "refund_horse_info", Length = 28, IsPrimaryKey = false, Comment = "返還馬番情報(馬番01～28): 0:返還なし 1:返還あり 発売後取消しとなり返還対象となった馬番のエリアに\"1\"を設定" },
                            new NormalFieldDefinition { Position = 68, Name = "refund_frame_info", Length = 8, IsPrimaryKey = false, Comment = "返還枠番情報(枠番1～8): 0:返還なし 1:返還あり 発売後取消しとなり返還対象となった枠番のエリアに\"1\"を設定" },
                            new NormalFieldDefinition { Position = 76, Name = "refund_same_frame_info", Length = 8, IsPrimaryKey = false, Comment = "返還同枠情報(枠番1～8): 0:返還なし 1:返還あり 発売後取消しとなり同枠のみ取消しとなった枠番のエリアに\"1\"を設定" },
                            new RepeatFieldDefinition
                            {
                                Position = 84,
                                RepeatCount = 28,
                                Length = 15,
                                Table = new TableDefinition
                                {
                                    Name = "tansho_votes",
                                    Comment = "単勝票数: 28頭立てまで考慮し繰返し28回 馬番昇順01～28",
                                    Fields = new List<FieldDefinition>
                                    {
                                        new NormalFieldDefinition { Position = 1, Name = "horse_number", Length = 2, IsPrimaryKey = false, Comment = "馬番: 該当馬番" },
                                        new NormalFieldDefinition { Position = 3, Name = "vote_count", Length = 11, IsPrimaryKey = false, Comment = "票数: 単位百円 ALL0:発売前取消し、発売票数なし スペース:登録なし" },
                                        new NormalFieldDefinition { Position = 14, Name = "popularity", Length = 2, IsPrimaryKey = false, Comment = "人気順: スペース:登録なし '--':発売前取消 '**':発売後取消" }
                                    }
                                }
                            },
                            new RepeatFieldDefinition
                            {
                                Position = 504,
                                RepeatCount = 28,
                                Length = 15,
                                Table = new TableDefinition
                                {
                                    Name = "fukusho_votes",
                                    Comment = "複勝票数: 28頭立てまで考慮し繰返し28回 馬番昇順01～28",
                                    Fields = new List<FieldDefinition>
                                    {
                                        new NormalFieldDefinition { Position = 1, Name = "horse_number", Length = 2, IsPrimaryKey = false, Comment = "馬番: 該当馬番" },
                                        new NormalFieldDefinition { Position = 3, Name = "vote_count", Length = 11, IsPrimaryKey = false, Comment = "票数: 単位百円 ALL0:発売前取消し、発売票数なし スペース:登録なし" },
                                        new NormalFieldDefinition { Position = 14, Name = "popularity", Length = 2, IsPrimaryKey = false, Comment = "人気順: スペース:登録なし '--':発売前取消 '**':発売後取消" }
                                    }
                                }
                            },
                            new RepeatFieldDefinition
                            {
                                Position = 924,
                                RepeatCount = 36,
                                Length = 15,
                                Table = new TableDefinition
                                {
                                    Name = "wakuren_votes",
                                    Comment = "枠連票数: 組番昇順 1-1～8-8",
                                    Fields = new List<FieldDefinition>
                                    {
                                        new NormalFieldDefinition { Position = 1, Name = "combination", Length = 2, IsPrimaryKey = false, Comment = "組番: 該当枠番" },
                                        new NormalFieldDefinition { Position = 3, Name = "vote_count", Length = 11, IsPrimaryKey = false, Comment = "票数: 単位百円 ALL0:発売前取消し、発売票数なし スペース:登録なし" },
                                        new NormalFieldDefinition { Position = 14, Name = "popularity", Length = 2, IsPrimaryKey = false, Comment = "人気順: スペース:登録なし '--':発売前取消 '**':発売後取消" }
                                    }
                                }
                            },
                            new RepeatFieldDefinition
                            {
                                Position = 1464,
                                RepeatCount = 153,
                                Length = 18,
                                Table = new TableDefinition
                                {
                                    Name = "umaren_votes",
                                    Comment = "馬連票数: 組番昇順 01-02～17-18",
                                    Fields = new List<FieldDefinition>
                                    {
                                        new NormalFieldDefinition { Position = 1, Name = "combination", Length = 4, IsPrimaryKey = false, Comment = "組番: 該当組番" },
                                        new NormalFieldDefinition { Position = 5, Name = "vote_count", Length = 11, IsPrimaryKey = false, Comment = "票数: 単位百円 ALL0:発売前取消し、発売票数なし スペース:登録なし" },
                                        new NormalFieldDefinition { Position = 16, Name = "popularity", Length = 3, IsPrimaryKey = false, Comment = "人気順: スペース:登録なし '---':発売前取消 '***':発売後取消" }
                                    }
                                }
                            },
                            new RepeatFieldDefinition
                            {
                                Position = 4218,
                                RepeatCount = 153,
                                Length = 18,
                                Table = new TableDefinition
                                {
                                    Name = "wide_votes",
                                    Comment = "ワイド票数: 組番昇順 01-02～17-18",
                                    Fields = new List<FieldDefinition>
                                    {
                                        new NormalFieldDefinition { Position = 1, Name = "combination", Length = 4, IsPrimaryKey = false, Comment = "組番: 該当組番" },
                                        new NormalFieldDefinition { Position = 5, Name = "vote_count", Length = 11, IsPrimaryKey = false, Comment = "票数: 単位百円 ALL0:発売前取消し、発売票数なし スペース:登録なし" },
                                        new NormalFieldDefinition { Position = 16, Name = "popularity", Length = 3, IsPrimaryKey = false, Comment = "人気順: スペース:登録なし '---':発売前取消 '***':発売後取消" }
                                    }
                                }
                            },
                            new RepeatFieldDefinition
                            {
                                Position = 6972,
                                RepeatCount = 306,
                                Length = 18,
                                Table = new TableDefinition
                                {
                                    Name = "umatan_votes",
                                    Comment = "馬単票数: 組番昇順 01-02～18-17",
                                    Fields = new List<FieldDefinition>
                                    {
                                        new NormalFieldDefinition { Position = 1, Name = "combination", Length = 4, IsPrimaryKey = false, Comment = "組番: 該当組番" },
                                        new NormalFieldDefinition { Position = 5, Name = "vote_count", Length = 11, IsPrimaryKey = false, Comment = "票数: 単位百円 ALL0:発売前取消し、発売票数なし スペース:登録なし" },
                                        new NormalFieldDefinition { Position = 16, Name = "popularity", Length = 3, IsPrimaryKey = false, Comment = "人気順: スペース:登録なし '---':発売前取消 '***':発売後取消" }
                                    }
                                }
                            },
                            new RepeatFieldDefinition
                            {
                                Position = 12480,
                                RepeatCount = 816,
                                Length = 20,
                                Table = new TableDefinition
                                {
                                    Name = "sanrenpuku_votes",
                                    Comment = "3連複票数: 組番昇順 01-02-03～16-17-18",
                                    Fields = new List<FieldDefinition>
                                    {
                                        new NormalFieldDefinition { Position = 1, Name = "combination", Length = 6, IsPrimaryKey = false, Comment = "組番: 該当組番" },
                                        new NormalFieldDefinition { Position = 7, Name = "vote_count", Length = 11, IsPrimaryKey = false, Comment = "票数: 単位百円 ALL0:発売前取消し、発売票数なし スペース:登録なし" },
                                        new NormalFieldDefinition { Position = 18, Name = "popularity", Length = 3, IsPrimaryKey = false, Comment = "人気順: スペース:登録なし '---':発売前取消 '***':発売後取消" }
                                    }
                                }
                            },
                            new NormalFieldDefinition { Position = 28800, Name = "tansho_total_votes", Length = 11, IsPrimaryKey = false, Comment = "単勝票数合計: 単位百円 単勝票数の合計（返還分票数を含む）" },
                            new NormalFieldDefinition { Position = 28811, Name = "fukusho_total_votes", Length = 11, IsPrimaryKey = false, Comment = "複勝票数合計: 単位百円 複勝票数の合計（返還分票数を含む）" },
                            new NormalFieldDefinition { Position = 28822, Name = "wakuren_total_votes", Length = 11, IsPrimaryKey = false, Comment = "枠連票数合計: 単位百円 枠連票数の合計（返還分票数を含む）" },
                            new NormalFieldDefinition { Position = 28833, Name = "umaren_total_votes", Length = 11, IsPrimaryKey = false, Comment = "馬連票数合計: 単位百円 馬連票数の合計（返還分票数を含む）" },
                            new NormalFieldDefinition { Position = 28844, Name = "wide_total_votes", Length = 11, IsPrimaryKey = false, Comment = "ワイド票数合計: 単位百円 ワイド票数の合計（返還分票数を含む）" },
                            new NormalFieldDefinition { Position = 28855, Name = "umatan_total_votes", Length = 11, IsPrimaryKey = false, Comment = "馬単票数合計: 単位百円 馬単票数の合計（返還分票数を含む）" },
                            new NormalFieldDefinition { Position = 28866, Name = "sanrenpuku_total_votes", Length = 11, IsPrimaryKey = false, Comment = "3連複票数合計: 単位百円 3連複票数の合計（返還分票数を含む）" },
                            new NormalFieldDefinition { Position = 28877, Name = "tansho_refund_votes", Length = 11, IsPrimaryKey = false, Comment = "単勝返還票数合計: 単位百円 単勝返還分票数の合計（合計票数から引くことで有効票数が求まる）" },
                            new NormalFieldDefinition { Position = 28888, Name = "fukusho_refund_votes", Length = 11, IsPrimaryKey = false, Comment = "複勝返還票数合計: 単位百円 複勝返還分票数の合計（合計票数から引くことで有効票数が求まる）" },
                            new NormalFieldDefinition { Position = 28899, Name = "wakuren_refund_votes", Length = 11, IsPrimaryKey = false, Comment = "枠連返還票数合計: 単位百円 枠連返還分票数の合計（合計票数から引くことで有効票数が求まる）" },
                            new NormalFieldDefinition { Position = 28910, Name = "umaren_refund_votes", Length = 11, IsPrimaryKey = false, Comment = "馬連返還票数合計: 単位百円 馬連返還分票数の合計（合計票数から引くことで有効票数が求まる）" },
                            new NormalFieldDefinition { Position = 28921, Name = "wide_refund_votes", Length = 11, IsPrimaryKey = false, Comment = "ワイド返還票数合計: 単位百円 ワイド返還分票数の合計（合計票数から引くことで有効票数が求まる）" },
                            new NormalFieldDefinition { Position = 28932, Name = "umatan_refund_votes", Length = 11, IsPrimaryKey = false, Comment = "馬単返還票数合計: 単位百円 馬単返還分票数の合計（合計票数から引くことで有効票数が求まる）" },
                            new NormalFieldDefinition { Position = 28943, Name = "sanrenpuku_refund_votes", Length = 11, IsPrimaryKey = false, Comment = "3連複返還票数合計: 単位百円 3連複返還分票数の合計（合計票数から引くことで有効票数が求まる）" }
                        }
                    },
                    creationDateField = new FieldDefinition { Position = 4, Length = 8 }
                });

            // 票数6（3連単）レコード定義
            recordDefinitions.Add(
                new RecordDefinition
                {
                    RecordTypeId = "H6",
                    Table = new TableDefinition
                    {
                        Name = "vote_count_6",
                        Comment = "票数6（3連単）レコード: 3連単の票数情報（前日売最終～確定）",
                        Fields = new List<FieldDefinition>
                        {
                            new NormalFieldDefinition { Position = 1, Name = "record_type_id", Length = 2, IsPrimaryKey = false, Comment = "レコード種別ID: 'H6' をセット" },
                            new NormalFieldDefinition { Position = 3, Name = "data_kubun", Length = 1, IsPrimaryKey = false, Comment = "データ区分: 2:前日売最終 4:確定 5:確定(月曜) 9:レース中止 0:該当レコード削除" },
                            new NormalFieldDefinition { Position = 12, Name = "kaisai_year", Length = 4, IsPrimaryKey = true, Comment = "開催年: 該当レース施行年 西暦4桁 yyyy形式" },
                            new NormalFieldDefinition { Position = 16, Name = "kaisai_month_day", Length = 4, IsPrimaryKey = true, Comment = "開催月日: 該当レース施行月日 各2桁 mmdd形式" },
                            new NormalFieldDefinition { Position = 20, Name = "track_code", Length = 2, IsPrimaryKey = true, Comment = "競馬場コード: 該当レース施行競馬場 <コード表 2001.競馬場コード>参照" },
                            new NormalFieldDefinition { Position = 22, Name = "kaisai_kai", Length = 2, IsPrimaryKey = true, Comment = "開催回[第N回]: 該当レース施行回 その競馬場でその年の何回目の開催かを示す" },
                            new NormalFieldDefinition { Position = 24, Name = "kaisai_day", Length = 2, IsPrimaryKey = true, Comment = "開催日目[N日目]: そのレース施行回で何日目の開催かを示す" },
                            new NormalFieldDefinition { Position = 26, Name = "race_number", Length = 2, IsPrimaryKey = true, Comment = "レース番号: 該当レース番号" },
                            new NormalFieldDefinition { Position = 28, Name = "registration_count", Length = 2, IsPrimaryKey = false, Comment = "登録頭数: 出馬表発表時の登録頭数" },
                            new NormalFieldDefinition { Position = 30, Name = "start_count", Length = 2, IsPrimaryKey = false, Comment = "出走頭数: 登録頭数から出走取消と競走除外･発走除外を除いた頭数" },
                            new NormalFieldDefinition { Position = 32, Name = "sale_flag_sanrentan", Length = 1, IsPrimaryKey = false, Comment = "発売フラグ 3連単: 0:発売なし 1:発売前取消 3:発売後取消 7:発売あり" },
                            new NormalFieldDefinition { Position = 33, Name = "refund_horse_info", Length = 18, IsPrimaryKey = false, Comment = "返還馬番情報(馬番01～18): 0:返還なし 1:返還あり 発売後取消しとなり返還対象となった馬番のエリアに\"1\"を設定" },
                            new RepeatFieldDefinition
                            {
                                Position = 51,
                                RepeatCount = 4896,
                                Length = 21,
                                Table = new TableDefinition
                                {
                                    Name = "sanrentan_votes",
                                    Comment = "3連単票数: 組番昇順 01-02-03～18-17-16",
                                    Fields = new List<FieldDefinition>
                                    {
                                        new NormalFieldDefinition { Position = 1, Name = "combination", Length = 6, IsPrimaryKey = false, Comment = "組番: 該当組番" },
                                        new NormalFieldDefinition { Position = 7, Name = "vote_count", Length = 11, IsPrimaryKey = false, Comment = "票数: 単位百円 ALL0:発売前取消し、発売票数なし スペース:登録なし" },
                                        new NormalFieldDefinition { Position = 18, Name = "popularity", Length = 4, IsPrimaryKey = false, Comment = "人気順: スペース:登録なし '----':発売前取消 '****':発売後取消" }
                                    }
                                }
                            },
                            new NormalFieldDefinition { Position = 102867, Name = "sanrentan_total_votes", Length = 11, IsPrimaryKey = false, Comment = "3連単票数合計: 単位百円 3連単票数の合計（返還分票数を含む）" },
                            new NormalFieldDefinition { Position = 102878, Name = "sanrentan_refund_votes", Length = 11, IsPrimaryKey = false, Comment = "3連単返還票数合計: 単位百円 3連単返還分票数の合計（合計票数から引くことで有効票数が求まる）" }
                        }
                    },
                    creationDateField = new FieldDefinition { Position = 4, Length = 8 }
                });

            // オッズ1（単複枠）レコード定義
            recordDefinitions.Add(
                new RecordDefinition
                {
                    RecordTypeId = "O1",
                    Table = new TableDefinition
                    {
                        Name = "odds_1_tan_fuku_waku",
                        Comment = "オッズ1（単複枠）レコード: 単勝・複勝・枠連のオッズ情報",
                        Fields = new List<FieldDefinition>
                        {
                            new NormalFieldDefinition { Position = 1, Name = "record_type_id", Length = 2, IsPrimaryKey = false, Comment = "レコード種別ID: 'O1' をセット" },
                            new NormalFieldDefinition { Position = 3, Name = "data_kubun", Length = 1, IsPrimaryKey = false, Comment = "データ区分: 1:中間 2:前日売最終 3:最終 4:確定 5:確定(月曜) 9:レース中止 0:該当レコード削除" },
                            new NormalFieldDefinition { Position = 12, Name = "kaisai_year", Length = 4, IsPrimaryKey = true, Comment = "開催年: 該当レース施行年 西暦4桁 yyyy形式" },
                            new NormalFieldDefinition { Position = 16, Name = "kaisai_month_day", Length = 4, IsPrimaryKey = true, Comment = "開催月日: 該当レース施行月日 各2桁 mmdd形式" },
                            new NormalFieldDefinition { Position = 20, Name = "track_code", Length = 2, IsPrimaryKey = true, Comment = "競馬場コード: 該当レース施行競馬場 <コード表 2001.競馬場コード>参照" },
                            new NormalFieldDefinition { Position = 22, Name = "kaisai_kai", Length = 2, IsPrimaryKey = true, Comment = "開催回[第N回]: 該当レース施行回 その競馬場でその年の何回目の開催かを示す" },
                            new NormalFieldDefinition { Position = 24, Name = "kaisai_day", Length = 2, IsPrimaryKey = true, Comment = "開催日目[N日目]: そのレース施行回で何日目の開催かを示す" },
                            new NormalFieldDefinition { Position = 26, Name = "race_number", Length = 2, IsPrimaryKey = true, Comment = "レース番号: 該当レース番号" },
                            new NormalFieldDefinition { Position = 28, Name = "announce_month_day_hour_minute", Length = 8, IsPrimaryKey = true, Comment = "発表月日時分: 月日時分各2桁 中間オッズのみ設定 時系列オッズを使用する場合のみキーとして設定" },
                            new NormalFieldDefinition { Position = 36, Name = "registration_count", Length = 2, IsPrimaryKey = false, Comment = "登録頭数: 出馬表発表時の登録頭数" },
                            new NormalFieldDefinition { Position = 38, Name = "start_count", Length = 2, IsPrimaryKey = false, Comment = "出走頭数: 登録頭数から出走取消と競走除外･発走除外を除いた頭数" },
                            new NormalFieldDefinition { Position = 40, Name = "sale_flag_tansho", Length = 1, IsPrimaryKey = false, Comment = "発売フラグ 単勝: 0:発売なし 1:発売前取消 3:発売後取消 7:発売あり" },
                            new NormalFieldDefinition { Position = 41, Name = "sale_flag_fukusho", Length = 1, IsPrimaryKey = false, Comment = "発売フラグ 複勝: 0:発売なし 1:発売前取消 3:発売後取消 7:発売あり" },
                            new NormalFieldDefinition { Position = 42, Name = "sale_flag_wakuren", Length = 1, IsPrimaryKey = false, Comment = "発売フラグ 枠連: 0:発売なし 1:発売前取消 3:発売後取消 7:発売あり" },
                            new NormalFieldDefinition { Position = 43, Name = "fukusho_chakubarai_key", Length = 1, IsPrimaryKey = false, Comment = "複勝着払キー: 0:複勝発売なし 2:2着まで払い 3:3着まで払い" },
                            new RepeatFieldDefinition
                            {
                                Position = 44,
                                RepeatCount = 28,
                                Length = 8,
                                Table = new TableDefinition
                                {
                                    Name = "tansho_odds",
                                    Comment = "単勝オッズ: 28頭立てまで考慮し繰返し28回 馬番昇順01～28",
                                    Fields = new List<FieldDefinition>
                                    {
                                        new NormalFieldDefinition { Position = 1, Name = "horse_number", Length = 2, IsPrimaryKey = false, Comment = "馬番: 該当馬番" },
                                        new NormalFieldDefinition { Position = 3, Name = "odds", Length = 4, IsPrimaryKey = false, Comment = "オッズ: 999.9倍で設定 9999:999.9倍以上 0000:無投票 ----:発売前取消 ****:発売後取消 スペース:登録なし" },
                                        new NormalFieldDefinition { Position = 7, Name = "popularity", Length = 2, IsPrimaryKey = false, Comment = "人気順: スペース:登録なし --:発売前取消 **:発売後取消 無投票の時は発売されている組合せの最大値を設定" }
                                    }
                                }
                            },
                            new RepeatFieldDefinition
                            {
                                Position = 268,
                                RepeatCount = 28,
                                Length = 12,
                                Table = new TableDefinition
                                {
                                    Name = "fukusho_odds",
                                    Comment = "複勝オッズ: 28頭立てまで考慮し繰返し28回 馬番昇順01～28",
                                    Fields = new List<FieldDefinition>
                                    {
                                        new NormalFieldDefinition { Position = 1, Name = "horse_number", Length = 2, IsPrimaryKey = false, Comment = "馬番: 該当馬番" },
                                        new NormalFieldDefinition { Position = 3, Name = "min_odds", Length = 4, IsPrimaryKey = false, Comment = "最低オッズ: 999.9倍で設定 2004年8月13日以前は99.9倍が最高値 0999:99.9倍以上 0000:無投票 ----:発売前取消 ****:発売後取消 スペース:登録なし" },
                                        new NormalFieldDefinition { Position = 7, Name = "max_odds", Length = 4, IsPrimaryKey = false, Comment = "最高オッズ: 999.9倍で設定 2004年8月13日以前は99.9倍が最高値 0999:99.9倍以上 0000:無投票 ----:発売前取消 ****:発売後取消 スペース:登録なし" },
                                        new NormalFieldDefinition { Position = 11, Name = "popularity", Length = 2, IsPrimaryKey = false, Comment = "人気順: スペース:登録なし --:発売前取消 **:発売後取消 無投票の時は発売されている組合せの最大値を設定" }
                                    }
                                }
                            },
                            new RepeatFieldDefinition
                            {
                                Position = 604,
                                RepeatCount = 36,
                                Length = 9,
                                Table = new TableDefinition
                                {
                                    Name = "wakuren_odds",
                                    Comment = "枠連オッズ: 組番昇順 1-1～8-8",
                                    Fields = new List<FieldDefinition>
                                    {
                                        new NormalFieldDefinition { Position = 1, Name = "combination", Length = 2, IsPrimaryKey = false, Comment = "組番: 該当枠番" },
                                        new NormalFieldDefinition { Position = 3, Name = "odds", Length = 5, IsPrimaryKey = false, Comment = "オッズ: 9999.9倍で設定 2004年8月13日以前は999.9倍が最高値 09999:999.9倍以上 00000:無投票 -----:発売前取消 *****:発売後取消 スペース:登録なし" },
                                        new NormalFieldDefinition { Position = 8, Name = "popularity", Length = 2, IsPrimaryKey = false, Comment = "人気順: スペース:登録なし --:発売前取消 **:発売後取消 無投票の時は発売されている組合せの最大値を設定" }
                                    }
                                }
                            },
                            new NormalFieldDefinition { Position = 928, Name = "tansho_total_votes", Length = 11, IsPrimaryKey = false, Comment = "単勝票数合計: 単位百円 単勝票数の合計（返還分票数を含む）" },
                            new NormalFieldDefinition { Position = 939, Name = "fukusho_total_votes", Length = 11, IsPrimaryKey = false, Comment = "複勝票数合計: 単位百円 複勝票数の合計（返還分票数を含む）" },
                            new NormalFieldDefinition { Position = 950, Name = "wakuren_total_votes", Length = 11, IsPrimaryKey = false, Comment = "枠連票数合計: 単位百円 枠連票数の合計（返還分票数を含む）" }
                        }
                    },
                    creationDateField = new FieldDefinition { Position = 4, Length = 8 }
                });

            // オッズ2（馬連）レコード定義
            recordDefinitions.Add(
                new RecordDefinition
                {
                    RecordTypeId = "O2",
                    Table = new TableDefinition
                    {
                        Name = "odds_2_umaren",
                        Comment = "オッズ2（馬連）レコード: 馬連のオッズ情報",
                        Fields = new List<FieldDefinition>
                        {
                            new NormalFieldDefinition { Position = 1, Name = "record_type_id", Length = 2, IsPrimaryKey = false, Comment = "レコード種別ID: 'O2' をセット" },
                            new NormalFieldDefinition { Position = 3, Name = "data_kubun", Length = 1, IsPrimaryKey = false, Comment = "データ区分: 1:中間 2:前日売最終 3:最終 4:確定 5:確定(月曜) 9:レース中止 0:該当レコード削除" },
                            new NormalFieldDefinition { Position = 12, Name = "kaisai_year", Length = 4, IsPrimaryKey = true, Comment = "開催年: 該当レース施行年 西暦4桁 yyyy形式" },
                            new NormalFieldDefinition { Position = 16, Name = "kaisai_month_day", Length = 4, IsPrimaryKey = true, Comment = "開催月日: 該当レース施行月日 各2桁 mmdd形式" },
                            new NormalFieldDefinition { Position = 20, Name = "track_code", Length = 2, IsPrimaryKey = true, Comment = "競馬場コード: 該当レース施行競馬場 <コード表 2001.競馬場コード>参照" },
                            new NormalFieldDefinition { Position = 22, Name = "kaisai_kai", Length = 2, IsPrimaryKey = true, Comment = "開催回[第N回]: 該当レース施行回 その競馬場でその年の何回目の開催かを示す" },
                            new NormalFieldDefinition { Position = 24, Name = "kaisai_day", Length = 2, IsPrimaryKey = true, Comment = "開催日目[N日目]: そのレース施行回で何日目の開催かを示す" },
                            new NormalFieldDefinition { Position = 26, Name = "race_number", Length = 2, IsPrimaryKey = true, Comment = "レース番号: 該当レース番号" },
                            new NormalFieldDefinition { Position = 28, Name = "announce_month_day_hour_minute", Length = 8, IsPrimaryKey = true, Comment = "発表月日時分: 月日時分各2桁 中間オッズのみ設定 時系列オッズを使用する場合のみキーとして設定" },
                            new NormalFieldDefinition { Position = 36, Name = "registration_count", Length = 2, IsPrimaryKey = false, Comment = "登録頭数: 出馬表発表時の登録頭数" },
                            new NormalFieldDefinition { Position = 38, Name = "start_count", Length = 2, IsPrimaryKey = false, Comment = "出走頭数: 登録頭数から出走取消と競走除外･発走除外を除いた頭数" },
                            new NormalFieldDefinition { Position = 40, Name = "sale_flag_umaren", Length = 1, IsPrimaryKey = false, Comment = "発売フラグ 馬連: 0:発売なし 1:発売前取消 3:発売後取消 7:発売あり" },
                            new RepeatFieldDefinition
                            {
                                Position = 41,
                                RepeatCount = 153,
                                Length = 13,
                                Table = new TableDefinition
                                {
                                    Name = "umaren_odds",
                                    Comment = "馬連オッズ: 組番昇順 01-02～17-18",
                                    Fields = new List<FieldDefinition>
                                    {
                                        new NormalFieldDefinition { Position = 1, Name = "combination", Length = 4, IsPrimaryKey = false, Comment = "組番: 該当組番" },
                                        new NormalFieldDefinition { Position = 5, Name = "odds", Length = 6, IsPrimaryKey = false, Comment = "オッズ: 99999.9倍で設定 2004年8月13日以前は9999.9倍が最高値 099999:9999.9倍以上 000000:無投票 ------:発売前取消 ******:発売後取消 スペース:登録なし" },
                                        new NormalFieldDefinition { Position = 11, Name = "popularity", Length = 3, IsPrimaryKey = false, Comment = "人気順: スペース:登録なし ---:発売前取消 ***:発売後取消 無投票の時は発売されている組合せの最大値を設定" }
                                    }
                                }
                            },
                            new NormalFieldDefinition { Position = 2030, Name = "umaren_total_votes", Length = 11, IsPrimaryKey = false, Comment = "馬連票数合計: 単位百円 馬連票数の合計（返還分票数を含む）" }
                        }
                    },
                    creationDateField = new FieldDefinition { Position = 4, Length = 8 }
                });

            // オッズ3（ワイド）レコード定義
            recordDefinitions.Add(
                new RecordDefinition
                {
                    RecordTypeId = "O3",
                    Table = new TableDefinition
                    {
                        Name = "odds_3_wide",
                        Comment = "オッズ3（ワイド）レコード: ワイドのオッズ情報",
                        Fields = new List<FieldDefinition>
                        {
                            new NormalFieldDefinition { Position = 1, Name = "record_type_id", Length = 2, IsPrimaryKey = false, Comment = "レコード種別ID: 'O3' をセット" },
                            new NormalFieldDefinition { Position = 3, Name = "data_kubun", Length = 1, IsPrimaryKey = false, Comment = "データ区分: 1:中間 2:前日売最終 3:最終 4:確定 5:確定(月曜) 9:レース中止 0:該当レコード削除" },
                            new NormalFieldDefinition { Position = 12, Name = "kaisai_year", Length = 4, IsPrimaryKey = true, Comment = "開催年: 該当レース施行年 西暦4桁 yyyy形式" },
                            new NormalFieldDefinition { Position = 16, Name = "kaisai_month_day", Length = 4, IsPrimaryKey = true, Comment = "開催月日: 該当レース施行月日 各2桁 mmdd形式" },
                            new NormalFieldDefinition { Position = 20, Name = "track_code", Length = 2, IsPrimaryKey = true, Comment = "競馬場コード: 該当レース施行競馬場 <コード表 2001.競馬場コード>参照" },
                            new NormalFieldDefinition { Position = 22, Name = "kaisai_kai", Length = 2, IsPrimaryKey = true, Comment = "開催回[第N回]: 該当レース施行回 その競馬場でその年の何回目の開催かを示す" },
                            new NormalFieldDefinition { Position = 24, Name = "kaisai_day", Length = 2, IsPrimaryKey = true, Comment = "開催日目[N日目]: そのレース施行回で何日目の開催かを示す" },
                            new NormalFieldDefinition { Position = 26, Name = "race_number", Length = 2, IsPrimaryKey = true, Comment = "レース番号: 該当レース番号" },
                            new NormalFieldDefinition { Position = 28, Name = "announce_month_day_hour_minute", Length = 8, IsPrimaryKey = true, Comment = "発表月日時分: 月日時分各2桁 中間オッズのみ設定 時系列オッズを使用する場合のみキーとして設定" },
                            new NormalFieldDefinition { Position = 36, Name = "registration_count", Length = 2, IsPrimaryKey = false, Comment = "登録頭数: 出馬表発表時の登録頭数" },
                            new NormalFieldDefinition { Position = 38, Name = "start_count", Length = 2, IsPrimaryKey = false, Comment = "出走頭数: 登録頭数から出走取消と競走除外･発走除外を除いた頭数" },
                            new NormalFieldDefinition { Position = 40, Name = "sale_flag_wide", Length = 1, IsPrimaryKey = false, Comment = "発売フラグ ワイド: 0:発売なし 1:発売前取消 3:発売後取消 7:発売あり" },
                            new RepeatFieldDefinition
                            {
                                Position = 41,
                                RepeatCount = 153,
                                Length = 17,
                                Table = new TableDefinition
                                {
                                    Name = "wide_odds",
                                    Comment = "ワイドオッズ: 組番昇順 01-02～17-18",
                                    Fields = new List<FieldDefinition>
                                    {
                                        new NormalFieldDefinition { Position = 1, Name = "combination", Length = 4, IsPrimaryKey = false, Comment = "組番: 該当組番" },
                                        new NormalFieldDefinition { Position = 5, Name = "min_odds", Length = 5, IsPrimaryKey = false, Comment = "最低オッズ: 9999.9倍で設定 99999:9999.9倍以上 00000:無投票 -----:発売前取消 *****:発売後取消 スペース:登録なし" },
                                        new NormalFieldDefinition { Position = 10, Name = "max_odds", Length = 5, IsPrimaryKey = false, Comment = "最高オッズ: 9999.9倍で設定 99999:9999.9倍以上 00000:無投票 -----:発売前取消 *****:発売後取消 スペース:登録なし" },
                                        new NormalFieldDefinition { Position = 15, Name = "popularity", Length = 3, IsPrimaryKey = false, Comment = "人気順: スペース:登録なし ---:発売前取消 ***:発売後取消 無投票の時は発売されている組合せの最大値を設定" }
                                    }
                                }
                            },
                            new NormalFieldDefinition { Position = 2642, Name = "wide_total_votes", Length = 11, IsPrimaryKey = false, Comment = "ワイド票数合計: 単位百円 ワイド票数の合計（返還分票数を含む）" }
                        }
                    },
                    creationDateField = new FieldDefinition { Position = 4, Length = 8 }
                });

            // オッズ4（馬単）レコード定義
            recordDefinitions.Add(
                new RecordDefinition
                {
                    RecordTypeId = "O4",
                    Table = new TableDefinition
                    {
                        Name = "odds_4_umatan",
                        Comment = "オッズ4（馬単）レコード: 馬単のオッズ情報",
                        Fields = new List<FieldDefinition>
                        {
                            new NormalFieldDefinition { Position = 1, Name = "record_type_id", Length = 2, IsPrimaryKey = false, Comment = "レコード種別ID: 'O4' をセット" },
                            new NormalFieldDefinition { Position = 3, Name = "data_kubun", Length = 1, IsPrimaryKey = false, Comment = "データ区分: 1:中間 2:前日売最終 3:最終 4:確定 5:確定(月曜) 9:レース中止 0:該当レコード削除" },
                            new NormalFieldDefinition { Position = 12, Name = "kaisai_year", Length = 4, IsPrimaryKey = true, Comment = "開催年: 該当レース施行年 西暦4桁 yyyy形式" },
                            new NormalFieldDefinition { Position = 16, Name = "kaisai_month_day", Length = 4, IsPrimaryKey = true, Comment = "開催月日: 該当レース施行月日 各2桁 mmdd形式" },
                            new NormalFieldDefinition { Position = 20, Name = "track_code", Length = 2, IsPrimaryKey = true, Comment = "競馬場コード: 該当レース施行競馬場 <コード表 2001.競馬場コード>参照" },
                            new NormalFieldDefinition { Position = 22, Name = "kaisai_kai", Length = 2, IsPrimaryKey = true, Comment = "開催回[第N回]: 該当レース施行回 その競馬場でその年の何回目の開催かを示す" },
                            new NormalFieldDefinition { Position = 24, Name = "kaisai_day", Length = 2, IsPrimaryKey = true, Comment = "開催日目[N日目]: そのレース施行回で何日目の開催かを示す" },
                            new NormalFieldDefinition { Position = 26, Name = "race_number", Length = 2, IsPrimaryKey = true, Comment = "レース番号: 該当レース番号" },
                            new NormalFieldDefinition { Position = 28, Name = "announce_month_day_hour_minute", Length = 8, IsPrimaryKey = true, Comment = "発表月日時分: 月日時分各2桁 中間オッズのみ設定 時系列オッズを使用する場合のみキーとして設定" },
                            new NormalFieldDefinition { Position = 36, Name = "registration_count", Length = 2, IsPrimaryKey = false, Comment = "登録頭数: 出馬表発表時の登録頭数" },
                            new NormalFieldDefinition { Position = 38, Name = "start_count", Length = 2, IsPrimaryKey = false, Comment = "出走頭数: 登録頭数から出走取消と競走除外･発走除外を除いた頭数" },
                            new NormalFieldDefinition { Position = 40, Name = "sale_flag_umatan", Length = 1, IsPrimaryKey = false, Comment = "発売フラグ 馬単: 0:発売なし 1:発売前取消 3:発売後取消 7:発売あり" },
                            new RepeatFieldDefinition
                            {
                                Position = 41,
                                RepeatCount = 306,
                                Length = 13,
                                Table = new TableDefinition
                                {
                                    Name = "umatan_odds",
                                    Comment = "馬単オッズ: 組番昇順 01-02～18-17",
                                    Fields = new List<FieldDefinition>
                                    {
                                        new NormalFieldDefinition { Position = 1, Name = "combination", Length = 4, IsPrimaryKey = false, Comment = "組番: 該当組番" },
                                        new NormalFieldDefinition { Position = 5, Name = "odds", Length = 6, IsPrimaryKey = false, Comment = "オッズ: 99999.9倍で設定 2004年8月13日以前は9999.9倍が最高値 099999:9999.9倍以上 000000:無投票 ------:発売前取消 ******:発売後取消 スペース:登録なし" },
                                        new NormalFieldDefinition { Position = 11, Name = "popularity", Length = 3, IsPrimaryKey = false, Comment = "人気順: スペース:登録なし ---:発売前取消 ***:発売後取消 無投票の時は発売されている組合せの最大値を設定" }
                                    }
                                }
                            },
                            new NormalFieldDefinition { Position = 4019, Name = "umatan_total_votes", Length = 11, IsPrimaryKey = false, Comment = "馬単票数合計: 単位百円 馬単票数の合計（返還分票数を含む）" }
                        }
                    },
                    creationDateField = new FieldDefinition { Position = 4, Length = 8 }
                });

            // オッズ5（3連複）レコード定義
            recordDefinitions.Add(
                new RecordDefinition
                {
                    RecordTypeId = "O5",
                    Table = new TableDefinition
                    {
                        Name = "odds_5_sanrenpuku",
                        Comment = "オッズ5（3連複）レコード: 3連複のオッズ情報",
                        Fields = new List<FieldDefinition>
                        {
                            new NormalFieldDefinition { Position = 1, Name = "record_type_id", Length = 2, IsPrimaryKey = false, Comment = "レコード種別ID: 'O5' をセット" },
                            new NormalFieldDefinition { Position = 3, Name = "data_kubun", Length = 1, IsPrimaryKey = false, Comment = "データ区分: 1:中間 2:前日売最終 3:最終 4:確定 5:確定(月曜) 9:レース中止 0:該当レコード削除" },
                            new NormalFieldDefinition { Position = 12, Name = "kaisai_year", Length = 4, IsPrimaryKey = true, Comment = "開催年: 該当レース施行年 西暦4桁 yyyy形式" },
                            new NormalFieldDefinition { Position = 16, Name = "kaisai_month_day", Length = 4, IsPrimaryKey = true, Comment = "開催月日: 該当レース施行月日 各2桁 mmdd形式" },
                            new NormalFieldDefinition { Position = 20, Name = "track_code", Length = 2, IsPrimaryKey = true, Comment = "競馬場コード: 該当レース施行競馬場 <コード表 2001.競馬場コード>参照" },
                            new NormalFieldDefinition { Position = 22, Name = "kaisai_kai", Length = 2, IsPrimaryKey = true, Comment = "開催回[第N回]: 該当レース施行回 その競馬場でその年の何回目の開催かを示す" },
                            new NormalFieldDefinition { Position = 24, Name = "kaisai_day", Length = 2, IsPrimaryKey = true, Comment = "開催日目[N日目]: そのレース施行回で何日目の開催かを示す" },
                            new NormalFieldDefinition { Position = 26, Name = "race_number", Length = 2, IsPrimaryKey = true, Comment = "レース番号: 該当レース番号" },
                            new NormalFieldDefinition { Position = 28, Name = "announce_month_day_hour_minute", Length = 8, IsPrimaryKey = true, Comment = "発表月日時分: 月日時分各2桁 中間オッズのみ設定 時系列オッズを使用する場合のみキーとして設定" },
                            new NormalFieldDefinition { Position = 36, Name = "registration_count", Length = 2, IsPrimaryKey = false, Comment = "登録頭数: 出馬表発表時の登録頭数" },
                            new NormalFieldDefinition { Position = 38, Name = "start_count", Length = 2, IsPrimaryKey = false, Comment = "出走頭数: 登録頭数から出走取消と競走除外･発走除外を除いた頭数" },
                            new NormalFieldDefinition { Position = 40, Name = "sale_flag_sanrenpuku", Length = 1, IsPrimaryKey = false, Comment = "発売フラグ 3連複: 0:発売なし 1:発売前取消 3:発売後取消 7:発売あり" },
                            new RepeatFieldDefinition
                            {
                                Position = 41,
                                RepeatCount = 816,
                                Length = 15,
                                Table = new TableDefinition
                                {
                                    Name = "sanrenpuku_odds",
                                    Comment = "3連複オッズ: 組番昇順 01-02-03～16-17-18",
                                    Fields = new List<FieldDefinition>
                                    {
                                        new NormalFieldDefinition { Position = 1, Name = "combination", Length = 6, IsPrimaryKey = false, Comment = "組番: 該当組番" },
                                        new NormalFieldDefinition { Position = 7, Name = "odds", Length = 6, IsPrimaryKey = false, Comment = "オッズ: 99999.9倍で設定 2004年8月13日以前は9999.9倍が最高値 099999:9999.9倍以上 000000:無投票 ------:発売前取消 ******:発売後取消 スペース:登録なし" },
                                        new NormalFieldDefinition { Position = 13, Name = "popularity", Length = 3, IsPrimaryKey = false, Comment = "人気順: スペース:登録なし ---:発売前取消 ***:発売後取消 無投票の時は発売されている組合せの最大値を設定" }
                                    }
                                }
                            },
                            new NormalFieldDefinition { Position = 12281, Name = "sanrenpuku_total_votes", Length = 11, IsPrimaryKey = false, Comment = "3連複票数合計: 単位百円 3連複票数の合計（返還分票数を含む）" }
                        }
                    },
                    creationDateField = new FieldDefinition { Position = 4, Length = 8 }
                });

            // オッズ6（3連単）レコード定義
            recordDefinitions.Add(
                new RecordDefinition
                {
                    RecordTypeId = "O6",
                    Table = new TableDefinition
                    {
                        Name = "odds_6_sanrentan",
                        Comment = "オッズ6（3連単）レコード: 3連単のオッズ情報",
                        Fields = new List<FieldDefinition>
                        {
                            new NormalFieldDefinition { Position = 1, Name = "record_type_id", Length = 2, IsPrimaryKey = false, Comment = "レコード種別ID: 'O6' をセット" },
                            new NormalFieldDefinition { Position = 3, Name = "data_kubun", Length = 1, IsPrimaryKey = false, Comment = "データ区分: 1:中間 2:前日売最終 3:最終 4:確定 5:確定(月曜) 9:レース中止 0:該当レコード削除" },
                            new NormalFieldDefinition { Position = 12, Name = "kaisai_year", Length = 4, IsPrimaryKey = true, Comment = "開催年: 該当レース施行年 西暦4桁 yyyy形式" },
                            new NormalFieldDefinition { Position = 16, Name = "kaisai_month_day", Length = 4, IsPrimaryKey = true, Comment = "開催月日: 該当レース施行月日 各2桁 mmdd形式" },
                            new NormalFieldDefinition { Position = 20, Name = "track_code", Length = 2, IsPrimaryKey = true, Comment = "競馬場コード: 該当レース施行競馬場 <コード表 2001.競馬場コード>参照" },
                            new NormalFieldDefinition { Position = 22, Name = "kaisai_kai", Length = 2, IsPrimaryKey = true, Comment = "開催回[第N回]: 該当レース施行回 その競馬場でその年の何回目の開催かを示す" },
                            new NormalFieldDefinition { Position = 24, Name = "kaisai_day", Length = 2, IsPrimaryKey = true, Comment = "開催日目[N日目]: そのレース施行回で何日目の開催かを示す" },
                            new NormalFieldDefinition { Position = 26, Name = "race_number", Length = 2, IsPrimaryKey = true, Comment = "レース番号: 該当レース番号" },
                            new NormalFieldDefinition { Position = 28, Name = "announce_month_day_hour_minute", Length = 8, IsPrimaryKey = true, Comment = "発表月日時分: 月日時分各2桁 中間オッズのみ設定 時系列オッズを使用する場合のみキーとして設定" },
                            new NormalFieldDefinition { Position = 36, Name = "registration_count", Length = 2, IsPrimaryKey = false, Comment = "登録頭数: 出馬表発表時の登録頭数" },
                            new NormalFieldDefinition { Position = 38, Name = "start_count", Length = 2, IsPrimaryKey = false, Comment = "出走頭数: 登録頭数から出走取消と競走除外･発走除外を除いた頭数" },
                            new NormalFieldDefinition { Position = 40, Name = "sale_flag_sanrentan", Length = 1, IsPrimaryKey = false, Comment = "発売フラグ 3連単: 0:発売なし 1:発売前取消 3:発売後取消 7:発売あり" },
                            new RepeatFieldDefinition
                            {
                                Position = 41,
                                RepeatCount = 4896,
                                Length = 17,
                                Table = new TableDefinition
                                {
                                    Name = "sanrentan_odds",
                                    Comment = "3連単オッズ: 組番昇順 01-02-03～18-17-16",
                                    Fields = new List<FieldDefinition>
                                    {
                                        new NormalFieldDefinition { Position = 1, Name = "combination", Length = 6, IsPrimaryKey = false, Comment = "組番: 該当組番" },
                                        new NormalFieldDefinition { Position = 7, Name = "odds", Length = 7, IsPrimaryKey = false, Comment = "オッズ: 999999.9倍で設定 0000000:無投票 -------:発売前取消 *******:発売後取消 スペース:登録なし" },
                                        new NormalFieldDefinition { Position = 14, Name = "popularity", Length = 4, IsPrimaryKey = false, Comment = "人気順: スペース:登録なし ----:発売前取消 ****:発売後取消 無投票の時は発売されている組合せの最大値を設定" }
                                    }
                                }
                            },
                            new NormalFieldDefinition { Position = 83273, Name = "sanrentan_total_votes", Length = 11, IsPrimaryKey = false, Comment = "3連単票数合計: 単位百円 3連単票数の合計（返還分票数を含む）" }
                        }
                    },
                    creationDateField = new FieldDefinition { Position = 4, Length = 8 }
                });

        }
    }
}

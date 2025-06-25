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
            public FieldDefinition CreationDateField { get; set; }
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
            public string DataType { get; set; } = "CHAR";
        }

        public class RepeatFieldDefinition : FieldDefinition
        {
            public int RepeatCount { get; set; }
            public TableDefinition Table { get; set; }
        }

        public class TableMetaData
        {
            public string TableName { get; set; }
            public List<NormalFieldDefinition> Columns { get; set; }
            public List<string> PrimaryKeys { get; set; }
            public string Comment { get; set; }
        }

        private readonly string[] commandLineArgs;
        private const string connectionString = "Host=localhost;Database=horgues3;Username=postgres;Password=postgres";
        private readonly List<RecordDefinition> recordDefinitions = new List<RecordDefinition>();
        private readonly Dictionary<string, Dictionary<string, Dictionary<string, Object>>> buffers = new Dictionary<string, Dictionary<string, Dictionary<string, Object>>>();
        private readonly List<TableMetaData> tableMetaData = new List<TableMetaData>();
        private const int batchSize = 10000;
        private const string sid = "SA000000/SD000004";

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
                        case "today":
                            ExecuteToday();
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

        private void ExecuteToday()
        {
            Console.WriteLine("Executing today process...");

            Console.WriteLine("Initializing JVLink...");
            int result = axJVLink1.JVInit(sid);
            if (result != 0)
            {
                throw new InvalidOperationException($"JVInit failed with error code: {result}");
            }
            Console.WriteLine("JVLink initialized successfully.");

            using (var connection = new NpgsqlConnection(connectionString))
            {
                connection.Open();

                string today = DateTime.Now.ToString("yyyyMMdd");
                Console.WriteLine($"Processing realtime data for date: {today}");

                // 開催日単位で取得するデータ種別ID
                string[] dailyDataSpecs = { "0B11", "0B12", "0B13", "0B14", "0B17", "0B51" };
                foreach (string dataSpec in dailyDataSpecs)
                {
                    Console.WriteLine($"Processing daily data: {dataSpec}");
                    ProcessRealtimeData(dataSpec, today, connection);
                }

                // レース毎に取得するデータ種別ID
                string[] raceDataSpecs = { "0B20", "0B30" };
                var raceIds = GetTodayRaceIds(connection, today);
                foreach (string raceId in raceIds)
                {
                    Console.WriteLine($"Processing race data for race: {raceId}");

                    foreach (string dataSpec in raceDataSpecs)
                    {
                        Console.WriteLine($"  Processing race data: {dataSpec}");
                        ProcessRealtimeData(dataSpec, raceId, connection);
                    }
                }

                // バッファに残っているデータをフラッシュ
                FlushAllBuffers(connection);
            }

            Console.WriteLine("Today process completed.");
        }

        private List<string> GetTodayRaceIds(NpgsqlConnection connection, string today)
        {
            var raceIds = new List<string>();

            // レースIDの取得クエリ
            string sql = @"
                SELECT DISTINCT 
                    kaisai_year || kaisai_monthday || keibajo_code || kaisai_kai || kaisai_nichime || race_number as race_id
                FROM race_shosai 
                WHERE kaisai_year || kaisai_monthday = @today
                ORDER BY race_id";
            using (var command = new NpgsqlCommand(sql, connection))
            {
                command.Parameters.AddWithValue("@today", today);

                using (var reader = command.ExecuteReader())
                {
                    while (reader.Read())
                    {
                        raceIds.Add(reader.GetString(reader.GetOrdinal("race_id")));
                    }
                }
            }

            Console.WriteLine($"Found {raceIds.Count} races for today");
            return raceIds;
        }

        private void ProcessRealtimeData(string dataSpec, string key, NpgsqlConnection connection)
        {
            Console.WriteLine($"Processing Realtime data...");
            Console.WriteLine($"  DataSpec: {dataSpec}");
            Console.WriteLine($"  Key: {key}");

            int result = axJVLink1.JVRTOpen(dataSpec, key);
            if (result == -1)
            {
                // 該当データなし - エラーではない
                Console.WriteLine($"No realtime data found for the specified criteria (DataSpec: {dataSpec}, Key: {key})");
                return;
            }
            else if (result != 0)
            {
                throw new InvalidOperationException($"JVRTOpen ({dataSpec}, {key}) failed with error code: {result}");
            }

            Console.WriteLine($"JVRTOpen executed successfully");

            // JVReadでデータを読み出し処理
            Console.WriteLine($"Starting data read and processing...");
            int totalRecords = 0;

            while (true)
            {
#pragma warning disable IDE0018
                int size = 110000;
#pragma warning restore IDE0018

                // JVReadでデータを読み込み
                result = axJVLink1.JVRead(out string buff, out size, out string filename);

                if (result == 0)
                {
                    // 全ファイル読み込み終了
                    Console.WriteLine("All files read completed.");
                    break;
                }
                else if (result == -1)
                {
                    // ファイル切り替わり
                    Console.WriteLine($"  File completed: {filename.Trim()}");
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

            Console.WriteLine($"Data processing completed. Total records: {totalRecords}");

            // バッファのフラッシュは呼び出し元で行うこと

            axJVLink1.JVClose();
            Console.WriteLine($"JV data processing completed.");

        }

        private void ProcessJVData(string dataSpec, string fromTime, int option)
        {
            Console.WriteLine($"Processing JV data...");
            Console.WriteLine($"  DataSpec: {dataSpec}");
            Console.WriteLine($"  FromTime: {fromTime}");
            Console.WriteLine($"  Option: {option}");

            int result;
            int filesToRead = 0, filesToDownload = 0;

            result = axJVLink1.JVOpen(dataSpec, fromTime, option, ref filesToRead, ref filesToDownload, out string lastFileTimestamp);
            if (result == -1)
            {
                // 該当データなし - エラーではない
                Console.WriteLine($"No data found for the specified criteria (DataSpec: {dataSpec}, FromTime: {fromTime})");
                return;
            }
            else if (result != 0)
            {
                throw new InvalidOperationException($"JVOpen ({dataSpec}) failed with error code: {result}");
            }

            Console.WriteLine($"JVOpen executed successfully:");
            Console.WriteLine($"  FilesToRead: {filesToRead}");
            Console.WriteLine($"  FilesToDownload: {filesToDownload}");
            Console.WriteLine($"  LastFileTimestamp: {lastFileTimestamp}");

            // JVStatusでダウンロード進捗を監視
            Console.WriteLine("Starting data download...");
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
                    Console.WriteLine($"  Downloaded files: {result}/{filesToDownload}");
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
#pragma warning disable IDE0018 // インライン変数宣言
                    int size = 110000;
#pragma warning restore IDE0018 // インライン変数宣言

                    // JVReadでデータを読み込み
                    result = axJVLink1.JVRead(out string buff, out size, out string filename);

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
                        Console.WriteLine($"  File completed: {filename.Trim()} (File {totalFiles})");
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

                // lastFileTimestampをデータベースに保存
                UpdateLastFileTimestamp(connection, lastFileTimestamp);
            }

            axJVLink1.JVClose();
            Console.WriteLine($"JV data processing completed.");
        }

        private void UpdateLastFileTimestamp(NpgsqlConnection connection, string lastFileTimestamp)
        {
            // 空文字列または無効な値の場合はスキップ
            if (string.IsNullOrWhiteSpace(lastFileTimestamp) || lastFileTimestamp.Length != 14)
            {
                Console.WriteLine($"Invalid lastFileTimestamp: {lastFileTimestamp}. Skipping update.");
                return;
            }

            // lastFileTimestampを更新または挿入（既存の値より新しい場合のみ）
            var upsertSql = @"
                INSERT INTO last_file_timestamp (id, last_file_timestamp, updated_at)
                VALUES (1, @lastFileTimestamp, CURRENT_TIMESTAMP)
                ON CONFLICT (id)
                DO UPDATE SET 
                    last_file_timestamp = EXCLUDED.last_file_timestamp,
                    updated_at = EXCLUDED.updated_at
                WHERE EXCLUDED.last_file_timestamp > last_file_timestamp.last_file_timestamp";

            using (var command = new NpgsqlCommand(upsertSql, connection))
            {
                command.Parameters.AddWithValue("@lastFileTimestamp", lastFileTimestamp);
                int rowsAffected = command.ExecuteNonQuery();

                if (rowsAffected > 0)
                {
                    Console.WriteLine($"LastFileTimestamp updated: {lastFileTimestamp}");
                }
                else
                {
                    Console.WriteLine($"LastFileTimestamp not updated (existing value is newer): {lastFileTimestamp}");
                }
            }
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

            string creationDate = System.Text.Encoding.GetEncoding("Shift_JIS").GetString(recordData, recordDefinition.CreationDateField.Position - 1, recordDefinition.CreationDateField.Length);

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
                // 親テーブル名がある場合は、親テーブル名を先頭に付加
                string inheritedKeyName = parentTableNames.Count > 0
                    ? $"{string.Join("_", parentTableNames)}_{parentKey.Key}"
                    : parentKey.Key;
                
                record[inheritedKeyName] = parentKey.Value;
                primaryKeys[inheritedKeyName] = parentKey.Value;
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
            Console.WriteLine("Executing setup process...");

            CreateTables();

            Console.WriteLine("Initializing JVLink...");
            int result = axJVLink1.JVInit(sid);
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
                Columns = new List<NormalFieldDefinition>(),
                PrimaryKeys = new List<string>(),
                Comment = table.Comment
            };
            
            var currentPrimaryKeys = new List<NormalFieldDefinition>();

            // 親テーブルの主キーを継承
            foreach (var pk in parentPrimaryKeys)
            {
                // 親テーブル名がある場合は、親テーブル名を先頭に付加したコピーを作成
                var inheritedPk = new NormalFieldDefinition
                {
                    Name = parentTableNames.Count > 0
                        ? $"{string.Join("_", parentTableNames)}_{pk.Name}"
                        : pk.Name,
                    DataType = pk.DataType,
                    Length = pk.Length,
                    IsPrimaryKey = pk.IsPrimaryKey,
                    Comment = pk.Comment,
                    Position = pk.Position
                };

                metadata.Columns.Add(inheritedPk);
                metadata.PrimaryKeys.Add(inheritedPk.Name);
                currentPrimaryKeys.Add(inheritedPk);
            }

            // 繰り返し番号を追加
            if (parentTableNames.Count > 0)
            {
                string indexColumnName = $"{fullTableName}_index";
                var indexColumn = new NormalFieldDefinition
                {
                    Name = indexColumnName,
                    DataType = "INTEGER",
                    Length = 0,
                    IsPrimaryKey = true,
                    Comment = "Repeat index"
                };
                metadata.Columns.Add(indexColumn);
                metadata.PrimaryKeys.Add(indexColumnName);
                currentPrimaryKeys.Add(indexColumn);
            }

            foreach (var field in table.Fields)
            {
                if (field is NormalFieldDefinition normalField)
                {
                    metadata.Columns.Add(normalField);
                    
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
            metadata.Columns.Add(new NormalFieldDefinition
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

        private void CreateLastFileTimestampTable(NpgsqlConnection connection)
        {
            var createTableSql = @"
                CREATE TABLE IF NOT EXISTS last_file_timestamp (
                    id INTEGER PRIMARY KEY DEFAULT 1,
                    last_file_timestamp CHAR(14) NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT single_row_check CHECK (id = 1)
                )";

            using (var command = new NpgsqlCommand(createTableSql, connection))
            {
                command.ExecuteNonQuery();
            }

            var commentSql = "COMMENT ON TABLE last_file_timestamp IS 'JVOpenで取得したlastFileTimestampを管理するテーブル'";
            using (var commentCommand = new NpgsqlCommand(commentSql, connection))
            {
                commentCommand.ExecuteNonQuery();
            }

            Console.WriteLine("  Table created: last_file_timestamp");
        }

        private void CreateTables()
        {
            Console.WriteLine("Creating database tables...");

            using (var connection = new NpgsqlConnection(connectionString))
            {
                connection.Open();

                // lastFileTimestamp管理テーブルの作成
                CreateLastFileTimestampTable(connection);

                // テーブルメタデータに基づいたテーブルの作成
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

                    Console.WriteLine($"  Table '{metadata.TableName}' created.");
                }
            }

            Console.WriteLine("Database tables created successfully.");
        }

        private string GetLastFileTimestamp()
        {
            using (var connection = new NpgsqlConnection(connectionString))
            {
                connection.Open();

                var selectSql = "SELECT last_file_timestamp FROM last_file_timestamp WHERE id = 1";
                using (var command = new NpgsqlCommand(selectSql, connection))
                {
                    var result = command.ExecuteScalar();

                    if (result != null)
                    {
                        string lastTimestamp = result.ToString();
                        Console.WriteLine($"Found existing last_file_timestamp: {lastTimestamp}");
                        return lastTimestamp;
                    }
                    else
                    {
                        throw new InvalidOperationException("last_file_timestamp record not found. Please run setup command first.");
                    }
                }
            }
        }

        private void ExecuteUpdate()
        {
            Console.WriteLine("Executing update process...");

            Console.WriteLine("Initializing JVLink...");
            int result = axJVLink1.JVInit(sid);
            if (result != 0)
            {
                throw new InvalidOperationException($"JVInit failed with error code: {result}");
            }
            Console.WriteLine("JVLink initialized successfully.");

            // データベースからlast_file_timestampを取得
            string fromTime = GetLastFileTimestamp();
            Console.WriteLine($"Retrieved last file timestamp from database: {fromTime}");

            // 全データ種別を一度に取得
            StringBuilder sb = new StringBuilder();
            sb.Append("TOKU");
            sb.Append("DIFN");
            sb.Append("HOSN");
            sb.Append("HOYU");
            sb.Append("COMM");
            sb.Append("RACE");
            sb.Append("SLOP");
            sb.Append("WOOD");
            sb.Append("YSCH");
            sb.Append("MING");
            sb.Append("BLDN");
            sb.Append("SNPN");
            string dataSpec = sb.ToString();
            ProcessJVData(dataSpec, fromTime, 1);

            Console.WriteLine("Update process completed.");
        }

        private void InitializeBuffers()
        {
            Console.WriteLine("Initializing data buffers...");
            
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
            // このメソッドは、recordDefinitions を作成するメソッドです
            // JV-Data仕様書に基づいてテーブル定義を作成します
            // 
            // 注意事項:
            // - 2項目以上の繰り返しはRepeatFieldDefinitionを使用してください
            // - 1項目の繰り返しは、RepeatFieldDefinitionは使わず、NormalFieldDefinitionを使用して 項目名_1, 項目名_2, ... のように定義してください
            // - テーブル名やカラム名は統一的な英語表記としてください
            //   例: 馬->uma, 騎手->kishu, 名->name, 番号->number, 年->year, 月日->monthday, 日/年月日->date,
            //       欧字->eng, 略称->short, 変更前->prev, 賞金->shokin, 前->mae, 後->ushiro, ハロン->furlong(s),
            //       数->su, 順位->juni, 馬番->umaban, 枠番->wakuban, 人気順->ninkijun, 組番->kumiban,
            //       馬齢->barei, 馬体重->bataiju, 時分->hourmin, 最低->min, 最高->max, 競走馬->kyosoba,
            //       生年月日->birth_date, 年月日場回日R->race_id, 新潟->nigata, 回数->kaisu, 時刻->jikoku
            //       重勝式->win5

            Console.WriteLine("Creating record definitions based on JV-Data specification...");
            recordDefinitions.Clear();




            // TK - 特別登録馬
            var tkRecord = new RecordDefinition
            {
                RecordTypeId = "TK",
                CreationDateField = new FieldDefinition { Position = 4, Length = 8 },
                Table = new TableDefinition
                {
                    Name = "tokubetsu_toroku_uma_joho",
                    Comment = "特別登録馬情報",
                    Fields = new List<FieldDefinition>
                    {
                        new NormalFieldDefinition { Position = 1, Length = 2, Name = "record_type_id", IsPrimaryKey = false, Comment = "レコード種別ID", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 3, Length = 1, Name = "data_kubun", IsPrimaryKey = false, Comment = "データ区分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 12, Length = 4, Name = "kaisai_year", IsPrimaryKey = true, Comment = "開催年", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 16, Length = 4, Name = "kaisai_monthday", IsPrimaryKey = true, Comment = "開催月日", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 20, Length = 2, Name = "keibajo_code", IsPrimaryKey = true, Comment = "競馬場コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 22, Length = 2, Name = "kaisai_kai", IsPrimaryKey = true, Comment = "開催回", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 24, Length = 2, Name = "kaisai_nichime", IsPrimaryKey = true, Comment = "開催日目", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 26, Length = 2, Name = "race_number", IsPrimaryKey = true, Comment = "レース番号", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 28, Length = 1, Name = "yobi_code", IsPrimaryKey = false, Comment = "曜日コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 29, Length = 4, Name = "tokubetsu_kyoso_number", IsPrimaryKey = false, Comment = "特別競走番号", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 33, Length = 60, Name = "kyoso_name_hondai", IsPrimaryKey = false, Comment = "競走名本題", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 93, Length = 60, Name = "kyoso_name_fukudai", IsPrimaryKey = false, Comment = "競走名副題", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 153, Length = 60, Name = "kyoso_name_kakko", IsPrimaryKey = false, Comment = "競走名カッコ内", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 213, Length = 120, Name = "kyoso_name_hondai_eng", IsPrimaryKey = false, Comment = "競走名本題欧字", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 333, Length = 120, Name = "kyoso_name_fukudai_eng", IsPrimaryKey = false, Comment = "競走名副題欧字", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 453, Length = 120, Name = "kyoso_name_kakko_eng", IsPrimaryKey = false, Comment = "競走名カッコ内欧字", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 573, Length = 20, Name = "kyoso_name_short_10", IsPrimaryKey = false, Comment = "競走名略称10文字", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 593, Length = 12, Name = "kyoso_name_short_6", IsPrimaryKey = false, Comment = "競走名略称6文字", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 605, Length = 6, Name = "kyoso_name_short_3", IsPrimaryKey = false, Comment = "競走名略称3文字", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 611, Length = 1, Name = "kyoso_name_kubun", IsPrimaryKey = false, Comment = "競走名区分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 612, Length = 3, Name = "jusho_kaiji", IsPrimaryKey = false, Comment = "重賞回次", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 615, Length = 1, Name = "grade_code", IsPrimaryKey = false, Comment = "グレードコード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 616, Length = 2, Name = "kyoso_shubetsu_code", IsPrimaryKey = false, Comment = "競走種別コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 618, Length = 3, Name = "kyoso_kigo_code", IsPrimaryKey = false, Comment = "競走記号コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 621, Length = 1, Name = "juryo_shubetsu_code", IsPrimaryKey = false, Comment = "重量種別コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 622, Length = 3, Name = "kyoso_joken_code_2sai", IsPrimaryKey = false, Comment = "競走条件コード2歳条件", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 625, Length = 3, Name = "kyoso_joken_code_3sai", IsPrimaryKey = false, Comment = "競走条件コード3歳条件", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 628, Length = 3, Name = "kyoso_joken_code_4sai", IsPrimaryKey = false, Comment = "競走条件コード4歳条件", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 631, Length = 3, Name = "kyoso_joken_code_5sai_ijo", IsPrimaryKey = false, Comment = "競走条件コード5歳以上条件", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 634, Length = 3, Name = "kyoso_joken_code_saijaku", IsPrimaryKey = false, Comment = "競走条件コード最若年条件", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 637, Length = 4, Name = "kyori", IsPrimaryKey = false, Comment = "距離", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 641, Length = 2, Name = "track_code", IsPrimaryKey = false, Comment = "トラックコード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 643, Length = 2, Name = "course_kubun", IsPrimaryKey = false, Comment = "コース区分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 645, Length = 8, Name = "hande_happyo_date", IsPrimaryKey = false, Comment = "ハンデ発表日", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 653, Length = 3, Name = "toroku_tosu", IsPrimaryKey = false, Comment = "登録頭数", DataType = "CHAR" },
                        
                        // 登録馬毎情報 (300回繰り返し)
                        new RepeatFieldDefinition
                        {
                            Position = 656,
                            Length = 70,
                            RepeatCount = 300,
                            Table = new TableDefinition
                            {
                                Name = "toroku_uma_goto_joho",
                                Comment = "登録馬毎情報",
                                Fields = new List<FieldDefinition>
                                {
                                    new NormalFieldDefinition { Position = 1, Length = 3, Name = "renban", IsPrimaryKey = false, Comment = "連番", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 4, Length = 10, Name = "ketto_toroku_number", IsPrimaryKey = false, Comment = "血統登録番号", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 14, Length = 36, Name = "uma_name", IsPrimaryKey = false, Comment = "馬名", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 50, Length = 2, Name = "uma_kigo_code", IsPrimaryKey = false, Comment = "馬記号コード", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 52, Length = 1, Name = "seibetsu_code", IsPrimaryKey = false, Comment = "性別コード", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 53, Length = 1, Name = "chokyoshi_tozai_shozoku_code", IsPrimaryKey = false, Comment = "調教師東西所属コード", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 54, Length = 5, Name = "chokyoshi_code", IsPrimaryKey = false, Comment = "調教師コード", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 59, Length = 8, Name = "chokyoshi_name_short", IsPrimaryKey = false, Comment = "調教師名略称", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 67, Length = 3, Name = "futan_juryo", IsPrimaryKey = false, Comment = "負担重量", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 70, Length = 1, Name = "koryu_kubun", IsPrimaryKey = false, Comment = "交流区分", DataType = "CHAR" }
                                }
                            }
                        }
                    }
                }
            };
            recordDefinitions.Add(tkRecord);

            // RA - レース詳細
            var raRecord = new RecordDefinition
            {
                RecordTypeId = "RA",
                CreationDateField = new FieldDefinition { Position = 4, Length = 8 },
                Table = new TableDefinition
                {
                    Name = "race_shosai",
                    Comment = "レース詳細",
                    Fields = new List<FieldDefinition>
                    {
                        new NormalFieldDefinition { Position = 1, Length = 2, Name = "record_type_id", IsPrimaryKey = false, Comment = "レコード種別ID", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 3, Length = 1, Name = "data_kubun", IsPrimaryKey = false, Comment = "データ区分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 12, Length = 4, Name = "kaisai_year", IsPrimaryKey = true, Comment = "開催年", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 16, Length = 4, Name = "kaisai_monthday", IsPrimaryKey = true, Comment = "開催月日", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 20, Length = 2, Name = "keibajo_code", IsPrimaryKey = true, Comment = "競馬場コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 22, Length = 2, Name = "kaisai_kai", IsPrimaryKey = true, Comment = "開催回", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 24, Length = 2, Name = "kaisai_nichime", IsPrimaryKey = true, Comment = "開催日目", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 26, Length = 2, Name = "race_number", IsPrimaryKey = true, Comment = "レース番号", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 28, Length = 1, Name = "yobi_code", IsPrimaryKey = false, Comment = "曜日コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 29, Length = 4, Name = "tokubetsu_kyoso_number", IsPrimaryKey = false, Comment = "特別競走番号", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 33, Length = 60, Name = "kyoso_name_hondai", IsPrimaryKey = false, Comment = "競走名本題", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 93, Length = 60, Name = "kyoso_name_fukudai", IsPrimaryKey = false, Comment = "競走名副題", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 153, Length = 60, Name = "kyoso_name_kakko", IsPrimaryKey = false, Comment = "競走名カッコ内", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 213, Length = 120, Name = "kyoso_name_hondai_eng", IsPrimaryKey = false, Comment = "競走名本題欧字", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 333, Length = 120, Name = "kyoso_name_fukudai_eng", IsPrimaryKey = false, Comment = "競走名副題欧字", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 453, Length = 120, Name = "kyoso_name_kakko_eng", IsPrimaryKey = false, Comment = "競走名カッコ内欧字", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 573, Length = 20, Name = "kyoso_name_short_10", IsPrimaryKey = false, Comment = "競走名略称10文字", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 593, Length = 12, Name = "kyoso_name_short_6", IsPrimaryKey = false, Comment = "競走名略称6文字", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 605, Length = 6, Name = "kyoso_name_short_3", IsPrimaryKey = false, Comment = "競走名略称3文字", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 611, Length = 1, Name = "kyoso_name_kubun", IsPrimaryKey = false, Comment = "競走名区分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 612, Length = 3, Name = "jusho_kaiji", IsPrimaryKey = false, Comment = "重賞回次", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 615, Length = 1, Name = "grade_code", IsPrimaryKey = false, Comment = "グレードコード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 616, Length = 1, Name = "prev_grade_code", IsPrimaryKey = false, Comment = "変更前グレードコード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 617, Length = 2, Name = "kyoso_shubetsu_code", IsPrimaryKey = false, Comment = "競走種別コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 619, Length = 3, Name = "kyoso_kigo_code", IsPrimaryKey = false, Comment = "競走記号コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 622, Length = 1, Name = "juryo_shubetsu_code", IsPrimaryKey = false, Comment = "重量種別コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 623, Length = 3, Name = "kyoso_joken_code_2sai", IsPrimaryKey = false, Comment = "競走条件コード2歳条件", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 626, Length = 3, Name = "kyoso_joken_code_3sai", IsPrimaryKey = false, Comment = "競走条件コード3歳条件", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 629, Length = 3, Name = "kyoso_joken_code_4sai", IsPrimaryKey = false, Comment = "競走条件コード4歳条件", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 632, Length = 3, Name = "kyoso_joken_code_5sai_ijo", IsPrimaryKey = false, Comment = "競走条件コード5歳以上条件", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 635, Length = 3, Name = "kyoso_joken_code_saijaku", IsPrimaryKey = false, Comment = "競走条件コード最若年条件", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 638, Length = 60, Name = "kyoso_joken_name", IsPrimaryKey = false, Comment = "競走条件名称", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 698, Length = 4, Name = "kyori", IsPrimaryKey = false, Comment = "距離", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 702, Length = 4, Name = "prev_kyori", IsPrimaryKey = false, Comment = "変更前距離", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 706, Length = 2, Name = "track_code", IsPrimaryKey = false, Comment = "トラックコード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 708, Length = 2, Name = "prev_track_code", IsPrimaryKey = false, Comment = "変更前トラックコード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 710, Length = 2, Name = "course_kubun", IsPrimaryKey = false, Comment = "コース区分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 712, Length = 2, Name = "prev_course_kubun", IsPrimaryKey = false, Comment = "変更前コース区分", DataType = "CHAR" },
                        
                        // 本賞金 (7回繰り返し)
                        new NormalFieldDefinition { Position = 714, Length = 8, Name = "hon_shokin_1", IsPrimaryKey = false, Comment = "本賞金1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 722, Length = 8, Name = "hon_shokin_2", IsPrimaryKey = false, Comment = "本賞金2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 730, Length = 8, Name = "hon_shokin_3", IsPrimaryKey = false, Comment = "本賞金3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 738, Length = 8, Name = "hon_shokin_4", IsPrimaryKey = false, Comment = "本賞金4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 746, Length = 8, Name = "hon_shokin_5", IsPrimaryKey = false, Comment = "本賞金5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 754, Length = 8, Name = "hon_shokin_6", IsPrimaryKey = false, Comment = "本賞金6着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 762, Length = 8, Name = "hon_shokin_7", IsPrimaryKey = false, Comment = "本賞金7着", DataType = "CHAR" },
                        
                        // 変更前本賞金 (5回繰り返し)
                        new NormalFieldDefinition { Position = 770, Length = 8, Name = "prev_hon_shokin_1", IsPrimaryKey = false, Comment = "変更前本賞金1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 778, Length = 8, Name = "prev_hon_shokin_2", IsPrimaryKey = false, Comment = "変更前本賞金2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 786, Length = 8, Name = "prev_hon_shokin_3", IsPrimaryKey = false, Comment = "変更前本賞金3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 794, Length = 8, Name = "prev_hon_shokin_4", IsPrimaryKey = false, Comment = "変更前本賞金4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 802, Length = 8, Name = "prev_hon_shokin_5", IsPrimaryKey = false, Comment = "変更前本賞金5着", DataType = "CHAR" },
                        
                        // 付加賞金 (5回繰り返し)
                        new NormalFieldDefinition { Position = 810, Length = 8, Name = "fuka_shokin_1", IsPrimaryKey = false, Comment = "付加賞金1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 818, Length = 8, Name = "fuka_shokin_2", IsPrimaryKey = false, Comment = "付加賞金2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 826, Length = 8, Name = "fuka_shokin_3", IsPrimaryKey = false, Comment = "付加賞金3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 834, Length = 8, Name = "fuka_shokin_4", IsPrimaryKey = false, Comment = "付加賞金4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 842, Length = 8, Name = "fuka_shokin_5", IsPrimaryKey = false, Comment = "付加賞金5着", DataType = "CHAR" },
                        
                        // 変更前付加賞金 (3回繰り返し)
                        new NormalFieldDefinition { Position = 850, Length = 8, Name = "prev_fuka_shokin_1", IsPrimaryKey = false, Comment = "変更前付加賞金1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 858, Length = 8, Name = "prev_fuka_shokin_2", IsPrimaryKey = false, Comment = "変更前付加賞金2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 866, Length = 8, Name = "prev_fuka_shokin_3", IsPrimaryKey = false, Comment = "変更前付加賞金3着", DataType = "CHAR" },
                        
                        new NormalFieldDefinition { Position = 874, Length = 4, Name = "hasso_jikoku", IsPrimaryKey = false, Comment = "発走時刻", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 878, Length = 4, Name = "prev_hasso_jikoku", IsPrimaryKey = false, Comment = "変更前発走時刻", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 882, Length = 2, Name = "toroku_tosu", IsPrimaryKey = false, Comment = "登録頭数", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 884, Length = 2, Name = "shusso_tosu", IsPrimaryKey = false, Comment = "出走頭数", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 886, Length = 2, Name = "nyusen_tosu", IsPrimaryKey = false, Comment = "入線頭数", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 888, Length = 1, Name = "tenko_code", IsPrimaryKey = false, Comment = "天候コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 889, Length = 1, Name = "shiba_baba_jotai_code", IsPrimaryKey = false, Comment = "芝馬場状態コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 890, Length = 1, Name = "dirt_baba_jotai_code", IsPrimaryKey = false, Comment = "ダート馬場状態コード", DataType = "CHAR" },
                        
                        // ラップタイム (25回繰り返し)
                        new NormalFieldDefinition { Position = 891, Length = 3, Name = "lap_time_1", IsPrimaryKey = false, Comment = "ラップタイム1", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 894, Length = 3, Name = "lap_time_2", IsPrimaryKey = false, Comment = "ラップタイム2", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 897, Length = 3, Name = "lap_time_3", IsPrimaryKey = false, Comment = "ラップタイム3", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 900, Length = 3, Name = "lap_time_4", IsPrimaryKey = false, Comment = "ラップタイム4", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 903, Length = 3, Name = "lap_time_5", IsPrimaryKey = false, Comment = "ラップタイム5", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 906, Length = 3, Name = "lap_time_6", IsPrimaryKey = false, Comment = "ラップタイム6", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 909, Length = 3, Name = "lap_time_7", IsPrimaryKey = false, Comment = "ラップタイム7", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 912, Length = 3, Name = "lap_time_8", IsPrimaryKey = false, Comment = "ラップタイム8", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 915, Length = 3, Name = "lap_time_9", IsPrimaryKey = false, Comment = "ラップタイム9", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 918, Length = 3, Name = "lap_time_10", IsPrimaryKey = false, Comment = "ラップタイム10", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 921, Length = 3, Name = "lap_time_11", IsPrimaryKey = false, Comment = "ラップタイム11", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 924, Length = 3, Name = "lap_time_12", IsPrimaryKey = false, Comment = "ラップタイム12", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 927, Length = 3, Name = "lap_time_13", IsPrimaryKey = false, Comment = "ラップタイム13", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 930, Length = 3, Name = "lap_time_14", IsPrimaryKey = false, Comment = "ラップタイム14", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 933, Length = 3, Name = "lap_time_15", IsPrimaryKey = false, Comment = "ラップタイム15", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 936, Length = 3, Name = "lap_time_16", IsPrimaryKey = false, Comment = "ラップタイム16", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 939, Length = 3, Name = "lap_time_17", IsPrimaryKey = false, Comment = "ラップタイム17", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 942, Length = 3, Name = "lap_time_18", IsPrimaryKey = false, Comment = "ラップタイム18", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 945, Length = 3, Name = "lap_time_19", IsPrimaryKey = false, Comment = "ラップタイム19", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 948, Length = 3, Name = "lap_time_20", IsPrimaryKey = false, Comment = "ラップタイム20", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 951, Length = 3, Name = "lap_time_21", IsPrimaryKey = false, Comment = "ラップタイム21", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 954, Length = 3, Name = "lap_time_22", IsPrimaryKey = false, Comment = "ラップタイム22", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 957, Length = 3, Name = "lap_time_23", IsPrimaryKey = false, Comment = "ラップタイム23", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 960, Length = 3, Name = "lap_time_24", IsPrimaryKey = false, Comment = "ラップタイム24", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 963, Length = 3, Name = "lap_time_25", IsPrimaryKey = false, Comment = "ラップタイム25", DataType = "CHAR" },
                        
                        new NormalFieldDefinition { Position = 966, Length = 4, Name = "shogai_mile_time", IsPrimaryKey = false, Comment = "障害マイルタイム", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 970, Length = 3, Name = "mae_3_furlongs", IsPrimaryKey = false, Comment = "前3ハロン", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 973, Length = 3, Name = "mae_4_furlongs", IsPrimaryKey = false, Comment = "前4ハロン", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 976, Length = 3, Name = "ushiro_3_furlongs", IsPrimaryKey = false, Comment = "後3ハロン", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 979, Length = 3, Name = "ushiro_4_furlongs", IsPrimaryKey = false, Comment = "後4ハロン", DataType = "CHAR" },
                        
                        // コーナー通過順位 (4回繰り返し)
                        new RepeatFieldDefinition
                        {
                            Position = 982,
                            Length = 72,
                            RepeatCount = 4,
                            Table = new TableDefinition
                            {
                                Name = "corner_tsuka_juni",
                                Comment = "コーナー通過順位",
                                Fields = new List<FieldDefinition>
                                {
                                    new NormalFieldDefinition { Position = 1, Length = 1, Name = "corner", IsPrimaryKey = false, Comment = "コーナー", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 2, Length = 1, Name = "shukai_su", IsPrimaryKey = false, Comment = "周回数", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 3, Length = 70, Name = "kaku_tsuka_juni", IsPrimaryKey = false, Comment = "各通過順位", DataType = "CHAR" }
                                }
                            }
                        },

                        new NormalFieldDefinition { Position = 1270, Length = 1, Name = "record_koshin_kubun", IsPrimaryKey = false, Comment = "レコード更新区分", DataType = "CHAR" }
                    }
                }
            };
            recordDefinitions.Add(raRecord);

            // SE - 馬毎レース情報
            var seRecord = new RecordDefinition
            {
                RecordTypeId = "SE",
                CreationDateField = new FieldDefinition { Position = 4, Length = 8 },
                Table = new TableDefinition
                {
                    Name = "uma_goto_race_joho",
                    Comment = "馬毎レース情報",
                    Fields = new List<FieldDefinition>
                    {
                        new NormalFieldDefinition { Position = 1, Length = 2, Name = "record_type_id", IsPrimaryKey = false, Comment = "レコード種別ID", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 3, Length = 1, Name = "data_kubun", IsPrimaryKey = false, Comment = "データ区分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 12, Length = 4, Name = "kaisai_year", IsPrimaryKey = true, Comment = "開催年", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 16, Length = 4, Name = "kaisai_monthday", IsPrimaryKey = true, Comment = "開催月日", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 20, Length = 2, Name = "keibajo_code", IsPrimaryKey = true, Comment = "競馬場コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 22, Length = 2, Name = "kaisai_kai", IsPrimaryKey = true, Comment = "開催回", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 24, Length = 2, Name = "kaisai_nichime", IsPrimaryKey = true, Comment = "開催日目", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 26, Length = 2, Name = "race_number", IsPrimaryKey = true, Comment = "レース番号", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 28, Length = 1, Name = "wakuban", IsPrimaryKey = false, Comment = "枠番", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 29, Length = 2, Name = "umaban", IsPrimaryKey = true, Comment = "馬番", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 31, Length = 10, Name = "ketto_toroku_number", IsPrimaryKey = true, Comment = "血統登録番号", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 41, Length = 36, Name = "uma_name", IsPrimaryKey = false, Comment = "馬名", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 77, Length = 2, Name = "uma_kigo_code", IsPrimaryKey = false, Comment = "馬記号コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 79, Length = 1, Name = "seibetsu_code", IsPrimaryKey = false, Comment = "性別コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 80, Length = 1, Name = "hinshu_code", IsPrimaryKey = false, Comment = "品種コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 81, Length = 2, Name = "keiro_code", IsPrimaryKey = false, Comment = "毛色コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 83, Length = 2, Name = "barei", IsPrimaryKey = false, Comment = "馬齢", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 85, Length = 1, Name = "tozai_shozoku_code", IsPrimaryKey = false, Comment = "東西所属コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 86, Length = 5, Name = "chokyoshi_code", IsPrimaryKey = false, Comment = "調教師コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 91, Length = 8, Name = "chokyoshi_name_short", IsPrimaryKey = false, Comment = "調教師名略称", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 99, Length = 6, Name = "banushi_code", IsPrimaryKey = false, Comment = "馬主コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 105, Length = 64, Name = "banushi_name", IsPrimaryKey = false, Comment = "馬主名(法人格無)", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 169, Length = 60, Name = "fukushoku_hyoji", IsPrimaryKey = false, Comment = "服色標示", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 229, Length = 60, Name = "yobi_1", IsPrimaryKey = false, Comment = "予備", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 289, Length = 3, Name = "futan_juryo", IsPrimaryKey = false, Comment = "負担重量", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 292, Length = 3, Name = "prev_futan_juryo", IsPrimaryKey = false, Comment = "変更前負担重量", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 295, Length = 1, Name = "blinker_shiyo_kubun", IsPrimaryKey = false, Comment = "ブリンカー使用区分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 296, Length = 1, Name = "yobi_2", IsPrimaryKey = false, Comment = "予備", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 297, Length = 5, Name = "kishu_code", IsPrimaryKey = false, Comment = "騎手コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 302, Length = 5, Name = "prev_kishu_code", IsPrimaryKey = false, Comment = "変更前騎手コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 307, Length = 8, Name = "kishu_name_short", IsPrimaryKey = false, Comment = "騎手名略称", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 315, Length = 8, Name = "prev_kishu_name_short", IsPrimaryKey = false, Comment = "変更前騎手名略称", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 323, Length = 1, Name = "kishu_minarai_code", IsPrimaryKey = false, Comment = "騎手見習コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 324, Length = 1, Name = "prev_kishu_minarai_code", IsPrimaryKey = false, Comment = "変更前騎手見習コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 325, Length = 3, Name = "bataiju", IsPrimaryKey = false, Comment = "馬体重", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 328, Length = 1, Name = "zogen_fugo", IsPrimaryKey = false, Comment = "増減符号", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 329, Length = 3, Name = "zogen_sa", IsPrimaryKey = false, Comment = "増減差", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 332, Length = 1, Name = "ijo_kubun_code", IsPrimaryKey = false, Comment = "異常区分コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 333, Length = 2, Name = "nyusen_juni", IsPrimaryKey = false, Comment = "入線順位", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 335, Length = 2, Name = "kakutei_chakujun", IsPrimaryKey = false, Comment = "確定着順", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 337, Length = 1, Name = "dochaku_kubun", IsPrimaryKey = false, Comment = "同着区分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 338, Length = 1, Name = "dochaku_tosu", IsPrimaryKey = false, Comment = "同着頭数", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 339, Length = 4, Name = "soha_time", IsPrimaryKey = false, Comment = "走破タイム", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 343, Length = 3, Name = "chakusa_code", IsPrimaryKey = false, Comment = "着差コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 346, Length = 3, Name = "plus_chakusa_code", IsPrimaryKey = false, Comment = "＋着差コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 349, Length = 3, Name = "plus_plus_chakusa_code", IsPrimaryKey = false, Comment = "＋＋着差コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 352, Length = 2, Name = "corner_juni_1", IsPrimaryKey = false, Comment = "1コーナーでの順位", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 354, Length = 2, Name = "corner_juni_2", IsPrimaryKey = false, Comment = "2コーナーでの順位", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 356, Length = 2, Name = "corner_juni_3", IsPrimaryKey = false, Comment = "3コーナーでの順位", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 358, Length = 2, Name = "corner_juni_4", IsPrimaryKey = false, Comment = "4コーナーでの順位", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 360, Length = 4, Name = "tansho_odds", IsPrimaryKey = false, Comment = "単勝オッズ", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 364, Length = 2, Name = "tansho_ninkijun", IsPrimaryKey = false, Comment = "単勝人気順", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 366, Length = 8, Name = "kakutoku_hon_shokin", IsPrimaryKey = false, Comment = "獲得本賞金", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 374, Length = 8, Name = "kakutoku_fuka_shokin", IsPrimaryKey = false, Comment = "獲得付加賞金", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 382, Length = 3, Name = "yobi_3", IsPrimaryKey = false, Comment = "予備", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 385, Length = 3, Name = "yobi_4", IsPrimaryKey = false, Comment = "予備", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 388, Length = 3, Name = "ushiro_4_furlongs_time", IsPrimaryKey = false, Comment = "後4ハロンタイム", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 391, Length = 3, Name = "ushiro_3_furlongs_time", IsPrimaryKey = false, Comment = "後3ハロンタイム", DataType = "CHAR" },
                        
                        // 1着馬(相手馬)情報 (3回繰り返し)
                        new RepeatFieldDefinition
                        {
                            Position = 394,
                            Length = 46,
                            RepeatCount = 3,
                            Table = new TableDefinition
                            {
                                Name = "aite_uma_joho",
                                Comment = "相手馬情報",
                                Fields = new List<FieldDefinition>
                                {
                                    new NormalFieldDefinition { Position = 1, Length = 10, Name = "ketto_toroku_number", IsPrimaryKey = false, Comment = "血統登録番号", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 11, Length = 36, Name = "uma_name", IsPrimaryKey = false, Comment = "馬名", DataType = "CHAR" }
                                }
                            }
                        },
                        
                        new NormalFieldDefinition { Position = 532, Length = 4, Name = "time_sa", IsPrimaryKey = false, Comment = "タイム差", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 536, Length = 1, Name = "record_koshin_kubun", IsPrimaryKey = false, Comment = "レコード更新区分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 537, Length = 1, Name = "mining_kubun", IsPrimaryKey = false, Comment = "マイニング区分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 538, Length = 5, Name = "mining_yoso_soha_time", IsPrimaryKey = false, Comment = "マイニング予想走破タイム", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 543, Length = 4, Name = "mining_yoso_gosa_plus", IsPrimaryKey = false, Comment = "マイニング予想誤差(信頼度)＋", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 547, Length = 4, Name = "mining_yoso_gosa_minus", IsPrimaryKey = false, Comment = "マイニング予想誤差(信頼度)－", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 551, Length = 2, Name = "mining_yoso_juni", IsPrimaryKey = false, Comment = "マイニング予想順位", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 553, Length = 1, Name = "konkai_race_kyakushitsu_hantei", IsPrimaryKey = false, Comment = "今回レース脚質判定", DataType = "CHAR" }
                    }
                }
            };
            recordDefinitions.Add(seRecord);

            // HR - 払戻情報
            var hrRecord = new RecordDefinition
            {
                RecordTypeId = "HR",
                CreationDateField = new FieldDefinition { Position = 4, Length = 8 },
                Table = new TableDefinition
                {
                    Name = "haraimodoshi_joho",
                    Comment = "払戻情報",
                    Fields = new List<FieldDefinition>
                    {
                        new NormalFieldDefinition { Position = 1, Length = 2, Name = "record_type_id", IsPrimaryKey = false, Comment = "レコード種別ID", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 3, Length = 1, Name = "data_kubun", IsPrimaryKey = false, Comment = "データ区分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 12, Length = 4, Name = "kaisai_year", IsPrimaryKey = true, Comment = "開催年", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 16, Length = 4, Name = "kaisai_monthday", IsPrimaryKey = true, Comment = "開催月日", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 20, Length = 2, Name = "keibajo_code", IsPrimaryKey = true, Comment = "競馬場コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 22, Length = 2, Name = "kaisai_kai", IsPrimaryKey = true, Comment = "開催回", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 24, Length = 2, Name = "kaisai_nichime", IsPrimaryKey = true, Comment = "開催日目", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 26, Length = 2, Name = "race_number", IsPrimaryKey = true, Comment = "レース番号", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 28, Length = 2, Name = "toroku_tosu", IsPrimaryKey = false, Comment = "登録頭数", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 30, Length = 2, Name = "shusso_tosu", IsPrimaryKey = false, Comment = "出走頭数", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 32, Length = 1, Name = "fuseirtsu_flag_tansho", IsPrimaryKey = false, Comment = "不成立フラグ単勝", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 33, Length = 1, Name = "fuseirtsu_flag_fukusho", IsPrimaryKey = false, Comment = "不成立フラグ複勝", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 34, Length = 1, Name = "fuseirtsu_flag_wakuren", IsPrimaryKey = false, Comment = "不成立フラグ枠連", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 35, Length = 1, Name = "fuseirtsu_flag_umaren", IsPrimaryKey = false, Comment = "不成立フラグ馬連", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 36, Length = 1, Name = "fuseirtsu_flag_wide", IsPrimaryKey = false, Comment = "不成立フラグワイド", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 37, Length = 1, Name = "yobi_1", IsPrimaryKey = false, Comment = "予備", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 38, Length = 1, Name = "fuseirtsu_flag_umatan", IsPrimaryKey = false, Comment = "不成立フラグ馬単", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 39, Length = 1, Name = "fuseirtsu_flag_sanrenpuku", IsPrimaryKey = false, Comment = "不成立フラグ3連複", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 40, Length = 1, Name = "fuseirtsu_flag_sanrentan", IsPrimaryKey = false, Comment = "不成立フラグ3連単", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 41, Length = 1, Name = "tokubarai_flag_tansho", IsPrimaryKey = false, Comment = "特払フラグ単勝", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 42, Length = 1, Name = "tokubarai_flag_fukusho", IsPrimaryKey = false, Comment = "特払フラグ複勝", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 43, Length = 1, Name = "tokubarai_flag_wakuren", IsPrimaryKey = false, Comment = "特払フラグ枠連", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 44, Length = 1, Name = "tokubarai_flag_umaren", IsPrimaryKey = false, Comment = "特払フラグ馬連", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 45, Length = 1, Name = "tokubarai_flag_wide", IsPrimaryKey = false, Comment = "特払フラグワイド", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 46, Length = 1, Name = "yobi_2", IsPrimaryKey = false, Comment = "予備", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 47, Length = 1, Name = "tokubarai_flag_umatan", IsPrimaryKey = false, Comment = "特払フラグ馬単", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 48, Length = 1, Name = "tokubarai_flag_sanrenpuku", IsPrimaryKey = false, Comment = "特払フラグ3連複", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 49, Length = 1, Name = "tokubarai_flag_sanrentan", IsPrimaryKey = false, Comment = "特払フラグ3連単", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 50, Length = 1, Name = "henkan_flag_tansho", IsPrimaryKey = false, Comment = "返還フラグ単勝", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 51, Length = 1, Name = "henkan_flag_fukusho", IsPrimaryKey = false, Comment = "返還フラグ複勝", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 52, Length = 1, Name = "henkan_flag_wakuren", IsPrimaryKey = false, Comment = "返還フラグ枠連", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 53, Length = 1, Name = "henkan_flag_umaren", IsPrimaryKey = false, Comment = "返還フラグ馬連", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 54, Length = 1, Name = "henkan_flag_wide", IsPrimaryKey = false, Comment = "返還フラグワイド", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 55, Length = 1, Name = "yobi_3", IsPrimaryKey = false, Comment = "予備", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 56, Length = 1, Name = "henkan_flag_umatan", IsPrimaryKey = false, Comment = "返還フラグ馬単", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 57, Length = 1, Name = "henkan_flag_sanrenpuku", IsPrimaryKey = false, Comment = "返還フラグ3連複", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 58, Length = 1, Name = "henkan_flag_sanrentan", IsPrimaryKey = false, Comment = "返還フラグ3連単", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 59, Length = 28, Name = "henkan_umaban_joho", IsPrimaryKey = false, Comment = "返還馬番情報(馬番01～28)", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 87, Length = 8, Name = "henkan_wakuban_joho", IsPrimaryKey = false, Comment = "返還枠番情報(枠番1～8)", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 95, Length = 8, Name = "henkan_dowaku_joho", IsPrimaryKey = false, Comment = "返還同枠情報(枠番1～8)", DataType = "CHAR" },
                        
                        // 単勝払戻 (3回繰り返し)
                        new RepeatFieldDefinition
                        {
                            Position = 103,
                            Length = 13,
                            RepeatCount = 3,
                            Table = new TableDefinition
                            {
                                Name = "tansho_haraimodoshi",
                                Comment = "単勝払戻",
                                Fields = new List<FieldDefinition>
                                {
                                    new NormalFieldDefinition { Position = 1, Length = 2, Name = "umaban", IsPrimaryKey = false, Comment = "馬番", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 3, Length = 9, Name = "haraimodoshi_kin", IsPrimaryKey = false, Comment = "払戻金", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 12, Length = 2, Name = "ninkijun", IsPrimaryKey = false, Comment = "人気順", DataType = "CHAR" }
                                }
                            }
                        },
                        
                        // 複勝払戻 (5回繰り返し)
                        new RepeatFieldDefinition
                        {
                            Position = 142,
                            Length = 13,
                            RepeatCount = 5,
                            Table = new TableDefinition
                            {
                                Name = "fukusho_haraimodoshi",
                                Comment = "複勝払戻",
                                Fields = new List<FieldDefinition>
                                {
                                    new NormalFieldDefinition { Position = 1, Length = 2, Name = "umaban", IsPrimaryKey = false, Comment = "馬番", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 3, Length = 9, Name = "haraimodoshi_kin", IsPrimaryKey = false, Comment = "払戻金", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 12, Length = 2, Name = "ninkijun", IsPrimaryKey = false, Comment = "人気順", DataType = "CHAR" }
                                }
                            }
                        },
                        
                        // 枠連払戻 (3回繰り返し)
                        new RepeatFieldDefinition
                        {
                            Position = 207,
                            Length = 13,
                            RepeatCount = 3,
                            Table = new TableDefinition
                            {
                                Name = "wakuren_haraimodoshi",
                                Comment = "枠連払戻",
                                Fields = new List<FieldDefinition>
                                {
                                    new NormalFieldDefinition { Position = 1, Length = 2, Name = "kumiban", IsPrimaryKey = false, Comment = "組番", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 3, Length = 9, Name = "haraimodoshi_kin", IsPrimaryKey = false, Comment = "払戻金", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 12, Length = 2, Name = "ninkijun", IsPrimaryKey = false, Comment = "人気順", DataType = "CHAR" }
                                }
                            }
                        },
                        
                        // 馬連払戻 (3回繰り返し)
                        new RepeatFieldDefinition
                        {
                            Position = 246,
                            Length = 16,
                            RepeatCount = 3,
                            Table = new TableDefinition
                            {
                                Name = "umaren_haraimodoshi",
                                Comment = "馬連払戻",
                                Fields = new List<FieldDefinition>
                                {
                                    new NormalFieldDefinition { Position = 1, Length = 4, Name = "kumiban", IsPrimaryKey = false, Comment = "組番", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 5, Length = 9, Name = "haraimodoshi_kin", IsPrimaryKey = false, Comment = "払戻金", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 14, Length = 3, Name = "ninkijun", IsPrimaryKey = false, Comment = "人気順", DataType = "CHAR" }
                                }
                            }
                        },
                        
                        // ワイド払戻 (7回繰り返し)
                        new RepeatFieldDefinition
                        {
                            Position = 294,
                            Length = 16,
                            RepeatCount = 7,
                            Table = new TableDefinition
                            {
                                Name = "wide_haraimodoshi",
                                Comment = "ワイド払戻",
                                Fields = new List<FieldDefinition>
                                {
                                    new NormalFieldDefinition { Position = 1, Length = 4, Name = "kumiban", IsPrimaryKey = false, Comment = "組番", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 5, Length = 9, Name = "haraimodoshi_kin", IsPrimaryKey = false, Comment = "払戻金", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 14, Length = 3, Name = "ninkijun", IsPrimaryKey = false, Comment = "人気順", DataType = "CHAR" }
                                }
                            }
                        },
                        
                        // 予備 (3回繰り返し)
                        new RepeatFieldDefinition
                        {
                            Position = 406,
                            Length = 16,
                            RepeatCount = 3,
                            Table = new TableDefinition
                            {
                                Name = "yobi_haraimodoshi",
                                Comment = "予備払戻",
                                Fields = new List<FieldDefinition>
                                {
                                    new NormalFieldDefinition { Position = 1, Length = 4, Name = "yobi_1", IsPrimaryKey = false, Comment = "予備", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 5, Length = 9, Name = "yobi_2", IsPrimaryKey = false, Comment = "予備", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 14, Length = 3, Name = "yobi_3", IsPrimaryKey = false, Comment = "予備", DataType = "CHAR" }
                                }
                            }
                        },
                        
                        // 馬単払戻 (6回繰り返し)
                        new RepeatFieldDefinition
                        {
                            Position = 454,
                            Length = 16,
                            RepeatCount = 6,
                            Table = new TableDefinition
                            {
                                Name = "umatan_haraimodoshi",
                                Comment = "馬単払戻",
                                Fields = new List<FieldDefinition>
                                {
                                    new NormalFieldDefinition { Position = 1, Length = 4, Name = "kumiban", IsPrimaryKey = false, Comment = "組番", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 5, Length = 9, Name = "haraimodoshi_kin", IsPrimaryKey = false, Comment = "払戻金", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 14, Length = 3, Name = "ninkijun", IsPrimaryKey = false, Comment = "人気順", DataType = "CHAR" }
                                }
                            }
                        },
                        
                        // 3連複払戻 (3回繰り返し)
                        new RepeatFieldDefinition
                        {
                            Position = 550,
                            Length = 18,
                            RepeatCount = 3,
                            Table = new TableDefinition
                            {
                                Name = "sanrenpuku_haraimodoshi",
                                Comment = "3連複払戻",
                                Fields = new List<FieldDefinition>
                                {
                                    new NormalFieldDefinition { Position = 1, Length = 6, Name = "kumiban", IsPrimaryKey = false, Comment = "組番", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 7, Length = 9, Name = "haraimodoshi_kin", IsPrimaryKey = false, Comment = "払戻金", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 16, Length = 3, Name = "ninkijun", IsPrimaryKey = false, Comment = "人気順", DataType = "CHAR" }
                                }
                            }
                        },
                        
                        // 3連単払戻 (6回繰り返し)
                        new RepeatFieldDefinition
                        {
                            Position = 604,
                            Length = 19,
                            RepeatCount = 6,
                            Table = new TableDefinition
                            {
                                Name = "sanrentan_haraimodoshi",
                                Comment = "3連単払戻",
                                Fields = new List<FieldDefinition>
                                {
                                    new NormalFieldDefinition { Position = 1, Length = 6, Name = "kumiban", IsPrimaryKey = false, Comment = "組番", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 7, Length = 9, Name = "haraimodoshi_kin", IsPrimaryKey = false, Comment = "払戻金", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 16, Length = 4, Name = "ninkijun", IsPrimaryKey = false, Comment = "人気順", DataType = "CHAR" }
                                }
                            }
                        }
                    }
                }
            };
            recordDefinitions.Add(hrRecord);

            // H1 - 票数１
            var h1Record = new RecordDefinition
            {
                RecordTypeId = "H1",
                CreationDateField = new FieldDefinition { Position = 4, Length = 8 },
                Table = new TableDefinition
                {
                    Name = "hyosu_1",
                    Comment = "票数１",
                    Fields = new List<FieldDefinition>
                    {
                        new NormalFieldDefinition { Position = 1, Length = 2, Name = "record_type_id", IsPrimaryKey = false, Comment = "レコード種別ID", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 3, Length = 1, Name = "data_kubun", IsPrimaryKey = false, Comment = "データ区分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 12, Length = 4, Name = "kaisai_year", IsPrimaryKey = true, Comment = "開催年", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 16, Length = 4, Name = "kaisai_monthday", IsPrimaryKey = true, Comment = "開催月日", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 20, Length = 2, Name = "keibajo_code", IsPrimaryKey = true, Comment = "競馬場コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 22, Length = 2, Name = "kaisai_kai", IsPrimaryKey = true, Comment = "開催回", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 24, Length = 2, Name = "kaisai_nichime", IsPrimaryKey = true, Comment = "開催日目", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 26, Length = 2, Name = "race_number", IsPrimaryKey = true, Comment = "レース番号", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 28, Length = 2, Name = "toroku_tosu", IsPrimaryKey = false, Comment = "登録頭数", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 30, Length = 2, Name = "shusso_tosu", IsPrimaryKey = false, Comment = "出走頭数", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 32, Length = 1, Name = "hatsubai_flag_tansho", IsPrimaryKey = false, Comment = "発売フラグ単勝", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 33, Length = 1, Name = "hatsubai_flag_fukusho", IsPrimaryKey = false, Comment = "発売フラグ複勝", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 34, Length = 1, Name = "hatsubai_flag_wakuren", IsPrimaryKey = false, Comment = "発売フラグ枠連", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 35, Length = 1, Name = "hatsubai_flag_umaren", IsPrimaryKey = false, Comment = "発売フラグ馬連", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 36, Length = 1, Name = "hatsubai_flag_wide", IsPrimaryKey = false, Comment = "発売フラグワイド", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 37, Length = 1, Name = "hatsubai_flag_umatan", IsPrimaryKey = false, Comment = "発売フラグ馬単", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 38, Length = 1, Name = "hatsubai_flag_sanrenpuku", IsPrimaryKey = false, Comment = "発売フラグ3連複", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 39, Length = 1, Name = "fukusho_chakubarai_key", IsPrimaryKey = false, Comment = "複勝着払キー", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 40, Length = 28, Name = "henkan_umaban_joho", IsPrimaryKey = false, Comment = "返還馬番情報", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 68, Length = 8, Name = "henkan_wakuban_joho", IsPrimaryKey = false, Comment = "返還枠番情報", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 76, Length = 8, Name = "henkan_dowaku_joho", IsPrimaryKey = false, Comment = "返還同枠情報", DataType = "CHAR" },
                        
                        // 単勝票数 (28回繰り返し)
                        new RepeatFieldDefinition
                        {
                            Position = 84,
                            Length = 15,
                            RepeatCount = 28,
                            Table = new TableDefinition
                            {
                                Name = "tansho_hyosu",
                                Comment = "単勝票数",
                                Fields = new List<FieldDefinition>
                                {
                                    new NormalFieldDefinition { Position = 1, Length = 2, Name = "umaban", IsPrimaryKey = false, Comment = "馬番", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 3, Length = 11, Name = "hyosu", IsPrimaryKey = false, Comment = "票数", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 14, Length = 2, Name = "ninkijun", IsPrimaryKey = false, Comment = "人気順", DataType = "CHAR" }
                                }
                            }
                        },
                        
                        // 複勝票数 (28回繰り返し)
                        new RepeatFieldDefinition
                        {
                            Position = 504,
                            Length = 15,
                            RepeatCount = 28,
                            Table = new TableDefinition
                            {
                                Name = "fukusho_hyosu",
                                Comment = "複勝票数",
                                Fields = new List<FieldDefinition>
                                {
                                    new NormalFieldDefinition { Position = 1, Length = 2, Name = "umaban", IsPrimaryKey = false, Comment = "馬番", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 3, Length = 11, Name = "hyosu", IsPrimaryKey = false, Comment = "票数", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 14, Length = 2, Name = "ninkijun", IsPrimaryKey = false, Comment = "人気順", DataType = "CHAR" }
                                }
                            }
                        },
                        
                        // 枠連票数 (36回繰り返し)
                        new RepeatFieldDefinition
                        {
                            Position = 924,
                            Length = 15,
                            RepeatCount = 36,
                            Table = new TableDefinition
                            {
                                Name = "wakuren_hyosu",
                                Comment = "枠連票数",
                                Fields = new List<FieldDefinition>
                                {
                                    new NormalFieldDefinition { Position = 1, Length = 2, Name = "kumiban", IsPrimaryKey = false, Comment = "組番", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 3, Length = 11, Name = "hyosu", IsPrimaryKey = false, Comment = "票数", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 14, Length = 2, Name = "ninkijun", IsPrimaryKey = false, Comment = "人気順", DataType = "CHAR" }
                                }
                            }
                        },
                        
                        // 馬連票数 (153回繰り返し)
                        new RepeatFieldDefinition
                        {
                            Position = 1464,
                            Length = 18,
                            RepeatCount = 153,
                            Table = new TableDefinition
                            {
                                Name = "umaren_hyosu",
                                Comment = "馬連票数",
                                Fields = new List<FieldDefinition>
                                {
                                    new NormalFieldDefinition { Position = 1, Length = 4, Name = "kumiban", IsPrimaryKey = false, Comment = "組番", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 5, Length = 11, Name = "hyosu", IsPrimaryKey = false, Comment = "票数", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 16, Length = 3, Name = "ninkijun", IsPrimaryKey = false, Comment = "人気順", DataType = "CHAR" }
                                }
                            }
                        },
                        
                        // ワイド票数 (153回繰り返し)
                        new RepeatFieldDefinition
                        {
                            Position = 4218,
                            Length = 18,
                            RepeatCount = 153,
                            Table = new TableDefinition
                            {
                                Name = "wide_hyosu",
                                Comment = "ワイド票数",
                                Fields = new List<FieldDefinition>
                                {
                                    new NormalFieldDefinition { Position = 1, Length = 4, Name = "kumiban", IsPrimaryKey = false, Comment = "組番", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 5, Length = 11, Name = "hyosu", IsPrimaryKey = false, Comment = "票数", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 16, Length = 3, Name = "ninkijun", IsPrimaryKey = false, Comment = "人気順", DataType = "CHAR" }
                                }
                            }
                        },
                        
                        // 馬単票数 (306回繰り返し)
                        new RepeatFieldDefinition
                        {
                            Position = 6972,
                            Length = 18,
                            RepeatCount = 306,
                            Table = new TableDefinition
                            {
                                Name = "umatan_hyosu",
                                Comment = "馬単票数",
                                Fields = new List<FieldDefinition>
                                {
                                    new NormalFieldDefinition { Position = 1, Length = 4, Name = "kumiban", IsPrimaryKey = false, Comment = "組番", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 5, Length = 11, Name = "hyosu", IsPrimaryKey = false, Comment = "票数", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 16, Length = 3, Name = "ninkijun", IsPrimaryKey = false, Comment = "人気順", DataType = "CHAR" }
                                }
                            }
                        },
                        
                        // 3連複票数 (816回繰り返し)
                        new RepeatFieldDefinition
                        {
                            Position = 12480,
                            Length = 20,
                            RepeatCount = 816,
                            Table = new TableDefinition
                            {
                                Name = "sanrenpuku_hyosu",
                                Comment = "3連複票数",
                                Fields = new List<FieldDefinition>
                                {
                                    new NormalFieldDefinition { Position = 1, Length = 6, Name = "kumiban", IsPrimaryKey = false, Comment = "組番", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 7, Length = 11, Name = "hyosu", IsPrimaryKey = false, Comment = "票数", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 18, Length = 3, Name = "ninkijun", IsPrimaryKey = false, Comment = "人気順", DataType = "CHAR" }
                                }
                            }
                        },
                        
                        // 各種票数合計
                        new NormalFieldDefinition { Position = 28800, Length = 11, Name = "tansho_hyosu_gokei", IsPrimaryKey = false, Comment = "単勝票数合計", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 28811, Length = 11, Name = "fukusho_hyosu_gokei", IsPrimaryKey = false, Comment = "複勝票数合計", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 28822, Length = 11, Name = "wakuren_hyosu_gokei", IsPrimaryKey = false, Comment = "枠連票数合計", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 28833, Length = 11, Name = "umaren_hyosu_gokei", IsPrimaryKey = false, Comment = "馬連票数合計", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 28844, Length = 11, Name = "wide_hyosu_gokei", IsPrimaryKey = false, Comment = "ワイド票数合計", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 28855, Length = 11, Name = "umatan_hyosu_gokei", IsPrimaryKey = false, Comment = "馬単票数合計", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 28866, Length = 11, Name = "sanrenpuku_hyosu_gokei", IsPrimaryKey = false, Comment = "3連複票数合計", DataType = "CHAR" },
                        
                        // 各種返還票数合計
                        new NormalFieldDefinition { Position = 28877, Length = 11, Name = "tansho_henkan_hyosu_gokei", IsPrimaryKey = false, Comment = "単勝返還票数合計", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 28888, Length = 11, Name = "fukusho_henkan_hyosu_gokei", IsPrimaryKey = false, Comment = "複勝返還票数合計", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 28899, Length = 11, Name = "wakuren_henkan_hyosu_gokei", IsPrimaryKey = false, Comment = "枠連返還票数合計", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 28910, Length = 11, Name = "umaren_henkan_hyosu_gokei", IsPrimaryKey = false, Comment = "馬連返還票数合計", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 28921, Length = 11, Name = "wide_henkan_hyosu_gokei", IsPrimaryKey = false, Comment = "ワイド返還票数合計", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 28932, Length = 11, Name = "umatan_henkan_hyosu_gokei", IsPrimaryKey = false, Comment = "馬単返還票数合計", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 28943, Length = 11, Name = "sanrenpuku_henkan_hyosu_gokei", IsPrimaryKey = false, Comment = "3連複返還票数合計", DataType = "CHAR" }
                    }
                }
            };
            recordDefinitions.Add(h1Record);

            // H6 - 票数6（3連単）
            var h6Record = new RecordDefinition
            {
                RecordTypeId = "H6",
                CreationDateField = new FieldDefinition { Position = 4, Length = 8 },
                Table = new TableDefinition
                {
                    Name = "hyosu_6",
                    Comment = "票数6（3連単）",
                    Fields = new List<FieldDefinition>
                    {
                        new NormalFieldDefinition { Position = 1, Length = 2, Name = "record_type_id", IsPrimaryKey = false, Comment = "レコード種別ID", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 3, Length = 1, Name = "data_kubun", IsPrimaryKey = false, Comment = "データ区分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 12, Length = 4, Name = "kaisai_year", IsPrimaryKey = true, Comment = "開催年", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 16, Length = 4, Name = "kaisai_monthday", IsPrimaryKey = true, Comment = "開催月日", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 20, Length = 2, Name = "keibajo_code", IsPrimaryKey = true, Comment = "競馬場コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 22, Length = 2, Name = "kaisai_kai", IsPrimaryKey = true, Comment = "開催回", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 24, Length = 2, Name = "kaisai_nichime", IsPrimaryKey = true, Comment = "開催日目", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 26, Length = 2, Name = "race_number", IsPrimaryKey = true, Comment = "レース番号", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 28, Length = 2, Name = "toroku_toshu", IsPrimaryKey = false, Comment = "登録頭数", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 30, Length = 2, Name = "shusso_toshu", IsPrimaryKey = false, Comment = "出走頭数", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 32, Length = 1, Name = "hatsubai_flag_sanrentan", IsPrimaryKey = false, Comment = "発売フラグ3連単", DataType = "CHAR" },
                        // 返還馬番情報(馬番01～18) - 18バイトの単一フィールド
                        new NormalFieldDefinition { Position = 33, Length = 18, Name = "henkan_umaban_joho", IsPrimaryKey = false, Comment = "返還馬番情報", DataType = "CHAR" },
                        // 3連単票数（繰り返し）
                        new RepeatFieldDefinition
                        {
                            Position = 51,
                            Length = 21,
                            RepeatCount = 4896,
                            Table = new TableDefinition
                            {
                                Name = "sanrentan_hyosu",
                                Comment = "3連単票数",
                                Fields = new List<FieldDefinition>
                                {
                                    new NormalFieldDefinition { Position = 1, Length = 6, Name = "kumiban", IsPrimaryKey = false, Comment = "組番", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 7, Length = 11, Name = "hyosu", IsPrimaryKey = false, Comment = "票数", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 18, Length = 4, Name = "ninkijun", IsPrimaryKey = false, Comment = "人気順", DataType = "CHAR" }
                                }
                            }
                        },
                        new NormalFieldDefinition { Position = 102867, Length = 11, Name = "sanrentan_hyosu_gokei", IsPrimaryKey = false, Comment = "3連単票数合計", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 102878, Length = 11, Name = "sanrentan_henkan_hyosu_gokei", IsPrimaryKey = false, Comment = "3連単返還票数合計", DataType = "CHAR" }
                    }
                }
            };
            recordDefinitions.Add(h6Record);

            // O1 - オッズ1（単複枠）
            var o1Record = new RecordDefinition
            {
                RecordTypeId = "O1",
                CreationDateField = new FieldDefinition { Position = 4, Length = 8 },
                Table = new TableDefinition
                {
                    Name = "odds_1",
                    Comment = "オッズ1（単複枠）",
                    Fields = new List<FieldDefinition>
                    {
                        new NormalFieldDefinition { Position = 1, Length = 2, Name = "record_type_id", IsPrimaryKey = false, Comment = "レコード種別ID", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 3, Length = 1, Name = "data_kubun", IsPrimaryKey = false, Comment = "データ区分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 12, Length = 4, Name = "kaisai_year", IsPrimaryKey = true, Comment = "開催年", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 16, Length = 4, Name = "kaisai_monthday", IsPrimaryKey = true, Comment = "開催月日", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 20, Length = 2, Name = "keibajo_code", IsPrimaryKey = true, Comment = "競馬場コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 22, Length = 2, Name = "kaisai_kai", IsPrimaryKey = true, Comment = "開催回", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 24, Length = 2, Name = "kaisai_nichime", IsPrimaryKey = true, Comment = "開催日目", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 26, Length = 2, Name = "race_number", IsPrimaryKey = true, Comment = "レース番号", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 28, Length = 8, Name = "happyo_monthday_hourmin", IsPrimaryKey = true, Comment = "発表月日時分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 36, Length = 2, Name = "toroku_tosu", IsPrimaryKey = false, Comment = "登録頭数", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 38, Length = 2, Name = "shusso_tosu", IsPrimaryKey = false, Comment = "出走頭数", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 40, Length = 1, Name = "hatsubai_flag_tansho", IsPrimaryKey = false, Comment = "発売フラグ単勝", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 41, Length = 1, Name = "hatsubai_flag_fukusho", IsPrimaryKey = false, Comment = "発売フラグ複勝", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 42, Length = 1, Name = "hatsubai_flag_wakuren", IsPrimaryKey = false, Comment = "発売フラグ枠連", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 43, Length = 1, Name = "fukusho_chakubarai_key", IsPrimaryKey = false, Comment = "複勝着払キー", DataType = "CHAR" },
                        new RepeatFieldDefinition
                        {
                            Position = 44,
                            Length = 8,
                            RepeatCount = 28,
                            Table = new TableDefinition
                            {
                                Name = "tansho_odds",
                                Comment = "単勝オッズ",
                                Fields = new List<FieldDefinition>
                                {
                                    new NormalFieldDefinition { Position = 1, Length = 2, Name = "umaban", IsPrimaryKey = false, Comment = "馬番", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 3, Length = 4, Name = "odds", IsPrimaryKey = false, Comment = "オッズ", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 7, Length = 2, Name = "ninkijun", IsPrimaryKey = false, Comment = "人気順", DataType = "CHAR" }
                                }
                            }
                        },
                        new RepeatFieldDefinition
                        {
                            Position = 268,
                            Length = 12,
                            RepeatCount = 28,
                            Table = new TableDefinition
                            {
                                Name = "fukusho_odds",
                                Comment = "複勝オッズ",
                                Fields = new List<FieldDefinition>
                                {
                                    new NormalFieldDefinition { Position = 1, Length = 2, Name = "umaban", IsPrimaryKey = false, Comment = "馬番", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 3, Length = 4, Name = "min_odds", IsPrimaryKey = false, Comment = "最低オッズ", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 7, Length = 4, Name = "max_odds", IsPrimaryKey = false, Comment = "最高オッズ", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 11, Length = 2, Name = "ninkijun", IsPrimaryKey = false, Comment = "人気順", DataType = "CHAR" }
                                }
                            }
                        },
                        new RepeatFieldDefinition
                        {
                            Position = 604,
                            Length = 9,
                            RepeatCount = 36,
                            Table = new TableDefinition
                            {
                                Name = "wakuren_odds",
                                Comment = "枠連オッズ",
                                Fields = new List<FieldDefinition>
                                {
                                    new NormalFieldDefinition { Position = 1, Length = 2, Name = "kumiban", IsPrimaryKey = false, Comment = "組番", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 3, Length = 5, Name = "odds", IsPrimaryKey = false, Comment = "オッズ", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 8, Length = 2, Name = "ninkijun", IsPrimaryKey = false, Comment = "人気順", DataType = "CHAR" }
                                }
                            }
                        },
                        new NormalFieldDefinition { Position = 928, Length = 11, Name = "tansho_hyosu_gokei", IsPrimaryKey = false, Comment = "単勝票数合計", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 939, Length = 11, Name = "fukusho_hyosu_gokei", IsPrimaryKey = false, Comment = "複勝票数合計", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 950, Length = 11, Name = "wakuren_hyosu_gokei", IsPrimaryKey = false, Comment = "枠連票数合計", DataType = "CHAR" }
                    }
                }
            };
            recordDefinitions.Add(o1Record);

            // O2 - オッズ2（馬連）
            var o2Record = new RecordDefinition
            {
                RecordTypeId = "O2",
                CreationDateField = new FieldDefinition { Position = 4, Length = 8 },
                Table = new TableDefinition
                {
                    Name = "odds_2",
                    Comment = "オッズ2（馬連）",
                    Fields = new List<FieldDefinition>
                    {
                        new NormalFieldDefinition { Position = 1, Length = 2, Name = "record_type_id", IsPrimaryKey = false, Comment = "レコード種別ID", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 3, Length = 1, Name = "data_kubun", IsPrimaryKey = false, Comment = "データ区分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 12, Length = 4, Name = "kaisai_year", IsPrimaryKey = true, Comment = "開催年", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 16, Length = 4, Name = "kaisai_monthday", IsPrimaryKey = true, Comment = "開催月日", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 20, Length = 2, Name = "keibajo_code", IsPrimaryKey = true, Comment = "競馬場コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 22, Length = 2, Name = "kaisai_kai", IsPrimaryKey = true, Comment = "開催回", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 24, Length = 2, Name = "kaisai_nichime", IsPrimaryKey = true, Comment = "開催日目", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 26, Length = 2, Name = "race_number", IsPrimaryKey = true, Comment = "レース番号", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 28, Length = 8, Name = "happyo_monthday_hourmin", IsPrimaryKey = true, Comment = "発表月日時分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 36, Length = 2, Name = "toroku_tosu", IsPrimaryKey = false, Comment = "登録頭数", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 38, Length = 2, Name = "shusso_tosu", IsPrimaryKey = false, Comment = "出走頭数", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 40, Length = 1, Name = "hatsubai_flag_umaren", IsPrimaryKey = false, Comment = "発売フラグ馬連", DataType = "CHAR" },
                        new RepeatFieldDefinition
                        {
                            Position = 41,
                            Length = 13,
                            RepeatCount = 153,
                            Table = new TableDefinition
                            {
                                Name = "umaren_odds",
                                Comment = "馬連オッズ",
                                Fields = new List<FieldDefinition>
                                {
                                    new NormalFieldDefinition { Position = 1, Length = 4, Name = "kumiban", IsPrimaryKey = false, Comment = "組番", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 5, Length = 6, Name = "odds", IsPrimaryKey = false, Comment = "オッズ", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 11, Length = 3, Name = "ninkijun", IsPrimaryKey = false, Comment = "人気順", DataType = "CHAR" }
                                }
                            }
                        },
                        new NormalFieldDefinition { Position = 2030, Length = 11, Name = "umaren_hyosu_goukei", IsPrimaryKey = false, Comment = "馬連票数合計", DataType = "CHAR" }
                    }
                }
            };
            recordDefinitions.Add(o2Record);

            // O3 - オッズ3（ワイド）
            var o3Record = new RecordDefinition
            {
                RecordTypeId = "O3",
                CreationDateField = new FieldDefinition { Position = 4, Length = 8 },
                Table = new TableDefinition
                {
                    Name = "odds_3",
                    Comment = "オッズ3（ワイド）",
                    Fields = new List<FieldDefinition>
                    {
                        new NormalFieldDefinition { Position = 1, Length = 2, Name = "record_type_id", IsPrimaryKey = false, Comment = "レコード種別ID", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 3, Length = 1, Name = "data_kubun", IsPrimaryKey = false, Comment = "データ区分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 12, Length = 4, Name = "kaisai_year", IsPrimaryKey = true, Comment = "開催年", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 16, Length = 4, Name = "kaisai_monthday", IsPrimaryKey = true, Comment = "開催月日", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 20, Length = 2, Name = "keibajo_code", IsPrimaryKey = true, Comment = "競馬場コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 22, Length = 2, Name = "kaisai_kai", IsPrimaryKey = true, Comment = "開催回", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 24, Length = 2, Name = "kaisai_nichime", IsPrimaryKey = true, Comment = "開催日目", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 26, Length = 2, Name = "race_number", IsPrimaryKey = true, Comment = "レース番号", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 28, Length = 8, Name = "happyo_monthday_hourmin", IsPrimaryKey = false, Comment = "発表月日時分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 36, Length = 2, Name = "toroku_tosu", IsPrimaryKey = false, Comment = "登録頭数", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 38, Length = 2, Name = "shusso_tosu", IsPrimaryKey = false, Comment = "出走頭数", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 40, Length = 1, Name = "hatsubai_flag_wide", IsPrimaryKey = false, Comment = "発売フラグワイド", DataType = "CHAR" },
                        new RepeatFieldDefinition
                        {
                            Position = 41,
                            Length = 17,
                            RepeatCount = 153,
                            Table = new TableDefinition
                            {
                                Name = "wide_odds",
                                Comment = "ワイドオッズ",
                                Fields = new List<FieldDefinition>
                                {
                                    new NormalFieldDefinition { Position = 1, Length = 4, Name = "kumiban", IsPrimaryKey = false, Comment = "組番", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 5, Length = 5, Name = "min_odds", IsPrimaryKey = false, Comment = "最低オッズ", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 10, Length = 5, Name = "max_odds", IsPrimaryKey = false, Comment = "最高オッズ", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 15, Length = 3, Name = "ninkijun", IsPrimaryKey = false, Comment = "人気順", DataType = "CHAR" }
                                }
                            }
                        },
                        new NormalFieldDefinition { Position = 2642, Length = 11, Name = "wide_hyosu_gokei", IsPrimaryKey = false, Comment = "ワイド票数合計", DataType = "CHAR" }
                    }
                }
            };
            recordDefinitions.Add(o3Record);

            // O4 - オッズ4（馬単）
            var o4Record = new RecordDefinition
            {
                RecordTypeId = "O4",
                CreationDateField = new FieldDefinition { Position = 4, Length = 8 },
                Table = new TableDefinition
                {
                    Name = "odds_4",
                    Comment = "馬単オッズ情報",
                    Fields = new List<FieldDefinition>
                    {
                        new NormalFieldDefinition { Position = 1, Length = 2, Name = "record_type_id", IsPrimaryKey = false, Comment = "レコード種別ID", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 3, Length = 1, Name = "data_kubun", IsPrimaryKey = false, Comment = "データ区分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 12, Length = 4, Name = "kaisai_year", IsPrimaryKey = true, Comment = "開催年", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 16, Length = 4, Name = "kaisai_monthday", IsPrimaryKey = true, Comment = "開催月日", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 20, Length = 2, Name = "keibajo_code", IsPrimaryKey = true, Comment = "競馬場コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 22, Length = 2, Name = "kaisai_kai", IsPrimaryKey = true, Comment = "開催回", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 24, Length = 2, Name = "kaisai_nichime", IsPrimaryKey = true, Comment = "開催日目", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 26, Length = 2, Name = "race_number", IsPrimaryKey = true, Comment = "レース番号", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 28, Length = 8, Name = "happyo_monthday_hourmin", IsPrimaryKey = false, Comment = "発表月日時分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 36, Length = 2, Name = "toroku_tosu", IsPrimaryKey = false, Comment = "登録頭数", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 38, Length = 2, Name = "shusso_tosu", IsPrimaryKey = false, Comment = "出走頭数", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 40, Length = 1, Name = "hatsubai_flag_umatan", IsPrimaryKey = false, Comment = "発売フラグ馬単", DataType = "CHAR" },
                        new RepeatFieldDefinition 
                        {
                            Position = 41,
                            Length = 13,
                            RepeatCount = 306,
                            Table = new TableDefinition
                            {
                                Name = "umatan_odds",
                                Comment = "馬単オッズ",
                                Fields = new List<FieldDefinition>
                                {
                                    new NormalFieldDefinition { Position = 1, Length = 4, Name = "kumiban", IsPrimaryKey = true, Comment = "組番", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 5, Length = 6, Name = "odds", IsPrimaryKey = false, Comment = "オッズ", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 11, Length = 3, Name = "ninkijun", IsPrimaryKey = false, Comment = "人気順", DataType = "CHAR" }
                                }
                            }
                        },
                        new NormalFieldDefinition { Position = 4019, Length = 11, Name = "umatan_hyosu_gokei", IsPrimaryKey = false, Comment = "馬単票数合計", DataType = "CHAR" }
                    }
                }
            };
            recordDefinitions.Add(o4Record);

            // O5 - オッズ5（3連複）
            var o5Record = new RecordDefinition
            {
                RecordTypeId = "O5",
                CreationDateField = new FieldDefinition { Position = 4, Length = 8 },
                Table = new TableDefinition
                {
                    Name = "odds_5",
                    Comment = "オッズ5（3連複）",
                    Fields = new List<FieldDefinition>
                    {
                        new NormalFieldDefinition { Position = 1, Length = 2, Name = "record_type_id", IsPrimaryKey = false, Comment = "レコード種別ID", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 3, Length = 1, Name = "data_kubun", IsPrimaryKey = false, Comment = "データ区分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 12, Length = 4, Name = "kaisai_year", IsPrimaryKey = true, Comment = "開催年", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 16, Length = 4, Name = "kaisai_monthday", IsPrimaryKey = true, Comment = "開催月日", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 20, Length = 2, Name = "keibajo_code", IsPrimaryKey = true, Comment = "競馬場コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 22, Length = 2, Name = "kaisai_kai", IsPrimaryKey = true, Comment = "開催回", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 24, Length = 2, Name = "kaisai_nichime", IsPrimaryKey = true, Comment = "開催日目", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 26, Length = 2, Name = "race_number", IsPrimaryKey = true, Comment = "レース番号", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 28, Length = 8, Name = "happyo_monthday_hourmin", IsPrimaryKey = false, Comment = "発表月日時分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 36, Length = 2, Name = "toroku_tosu", IsPrimaryKey = false, Comment = "登録頭数", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 38, Length = 2, Name = "shusso_tosu", IsPrimaryKey = false, Comment = "出走頭数", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 40, Length = 1, Name = "hatsubai_flag_sanrenpuku", IsPrimaryKey = false, Comment = "発売フラグ3連複", DataType = "CHAR" },
                        new RepeatFieldDefinition
                        {
                            Position = 41,
                            Length = 15,
                            RepeatCount = 816,
                            Table = new TableDefinition
                            {
                                Name = "sanrenpuku_odds",
                                Comment = "3連複オッズ",
                                Fields = new List<FieldDefinition>
                                {
                                    new NormalFieldDefinition { Position = 1, Length = 6, Name = "kumiban", IsPrimaryKey = false, Comment = "組番", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 7, Length = 6, Name = "odds", IsPrimaryKey = false, Comment = "オッズ", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 13, Length = 3, Name = "ninkijun", IsPrimaryKey = false, Comment = "人気順", DataType = "CHAR" }
                                }
                            }
                        },
                        new NormalFieldDefinition { Position = 12281, Length = 11, Name = "sanrenpuku_hyosu_goukei", IsPrimaryKey = false, Comment = "3連複票数合計", DataType = "CHAR" }
                    }
                }
            };
            recordDefinitions.Add(o5Record);

            // O6 - オッズ6（3連単）
            var o6Record = new RecordDefinition
            {
                RecordTypeId = "O6",
                CreationDateField = new FieldDefinition { Position = 4, Length = 8 },
                Table = new TableDefinition
                {
                    Name = "odds_6",
                    Comment = "オッズ6（3連単）",
                    Fields = new List<FieldDefinition>
                    {
                        new NormalFieldDefinition { Position = 1, Length = 2, Name = "record_type_id", IsPrimaryKey = false, Comment = "レコード種別ID", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 3, Length = 1, Name = "data_kubun", IsPrimaryKey = false, Comment = "データ区分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 12, Length = 4, Name = "kaisai_year", IsPrimaryKey = true, Comment = "開催年", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 16, Length = 4, Name = "kaisai_monthday", IsPrimaryKey = true, Comment = "開催月日", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 20, Length = 2, Name = "keibajo_code", IsPrimaryKey = true, Comment = "競馬場コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 22, Length = 2, Name = "kaisai_kai", IsPrimaryKey = true, Comment = "開催回", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 24, Length = 2, Name = "kaisai_nichime", IsPrimaryKey = true, Comment = "開催日目", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 26, Length = 2, Name = "race_number", IsPrimaryKey = true, Comment = "レース番号", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 28, Length = 8, Name = "happyo_monthday_hourmin", IsPrimaryKey = false, Comment = "発表月日時分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 36, Length = 2, Name = "toroku_tosu", IsPrimaryKey = false, Comment = "登録頭数", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 38, Length = 2, Name = "shusso_tosu", IsPrimaryKey = false, Comment = "出走頭数", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 40, Length = 1, Name = "hatsubai_flag_sanrentan", IsPrimaryKey = false, Comment = "発売フラグ3連単", DataType = "CHAR" },
                        new RepeatFieldDefinition
                        {
                            Position = 41,
                            Length = 17,
                            RepeatCount = 4896,
                            Table = new TableDefinition
                            {
                                Name = "sanrentan_odds",
                                Comment = "3連単オッズ",
                                Fields = new List<FieldDefinition>
                                {
                                    new NormalFieldDefinition { Position = 1, Length = 6, Name = "kumiban", IsPrimaryKey = false, Comment = "組番", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 7, Length = 7, Name = "odds", IsPrimaryKey = false, Comment = "オッズ", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 14, Length = 4, Name = "ninkijun", IsPrimaryKey = false, Comment = "人気順", DataType = "CHAR" }
                                }
                            }
                        },
                        new NormalFieldDefinition { Position = 83273, Length = 11, Name = "sanrentan_hyosu_gokei", IsPrimaryKey = false, Comment = "3連単票数合計", DataType = "CHAR" }
                    }
                }
            };
            recordDefinitions.Add(o6Record);

            // UM - 競走馬マスタ
            var umRecord = new RecordDefinition
            {
                RecordTypeId = "UM",
                CreationDateField = new FieldDefinition { Position = 4, Length = 8 },
                Table = new TableDefinition
                {
                    Name = "kyosoba_master",
                    Comment = "競走馬マスタ",
                    Fields = new List<FieldDefinition>
                    {
                        new NormalFieldDefinition { Position = 1, Length = 2, Name = "record_type_id", IsPrimaryKey = false, Comment = "レコード種別ID", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 3, Length = 1, Name = "data_kubun", IsPrimaryKey = false, Comment = "データ区分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 12, Length = 10, Name = "ketto_toroku_number", IsPrimaryKey = true, Comment = "血統登録番号", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 22, Length = 1, Name = "kyosoba_massho_kubun", IsPrimaryKey = false, Comment = "競走馬抹消区分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 23, Length = 8, Name = "kyosoba_toroku_date", IsPrimaryKey = false, Comment = "競走馬登録年月日", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 31, Length = 8, Name = "kyosoba_massho_date", IsPrimaryKey = false, Comment = "競走馬抹消年月日", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 39, Length = 8, Name = "birth_date", IsPrimaryKey = false, Comment = "生年月日", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 47, Length = 36, Name = "uma_name", IsPrimaryKey = false, Comment = "馬名", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 83, Length = 36, Name = "uma_name_hankaku_kana", IsPrimaryKey = false, Comment = "馬名半角カナ", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 119, Length = 60, Name = "uma_name_eng", IsPrimaryKey = false, Comment = "馬名欧字", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 179, Length = 1, Name = "jra_shisetsu_zaikyu_flag", IsPrimaryKey = false, Comment = "JRA施設在きゅうフラグ", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 180, Length = 19, Name = "yobi", IsPrimaryKey = false, Comment = "予備", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 199, Length = 2, Name = "uma_kigo_code", IsPrimaryKey = false, Comment = "馬記号コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 201, Length = 1, Name = "seibetsu_code", IsPrimaryKey = false, Comment = "性別コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 202, Length = 1, Name = "hinshu_code", IsPrimaryKey = false, Comment = "品種コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 203, Length = 2, Name = "keiro_code", IsPrimaryKey = false, Comment = "毛色コード", DataType = "CHAR" },
                        
                        // 3代血統情報 (14回繰り返し)
                        new RepeatFieldDefinition
                        {
                            Position = 205,
                            Length = 46,
                            RepeatCount = 14,
                            Table = new TableDefinition
                            {
                                Name = "sandai_ketto_joho",
                                Comment = "3代血統情報",
                                Fields = new List<FieldDefinition>
                                {
                                    new NormalFieldDefinition { Position = 1, Length = 10, Name = "hanshoku_toroku_number", IsPrimaryKey = false, Comment = "繁殖登録番号", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 11, Length = 36, Name = "uma_name", IsPrimaryKey = false, Comment = "馬名", DataType = "CHAR" }
                                }
                            }
                        },
                        
                        new NormalFieldDefinition { Position = 849, Length = 1, Name = "tozai_shozoku_code", IsPrimaryKey = false, Comment = "東西所属コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 850, Length = 5, Name = "chokyoshi_code", IsPrimaryKey = false, Comment = "調教師コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 855, Length = 8, Name = "chokyoshi_name_short", IsPrimaryKey = false, Comment = "調教師名略称", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 863, Length = 20, Name = "shotai_chiki_name", IsPrimaryKey = false, Comment = "招待地域名", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 883, Length = 8, Name = "seisansha_code", IsPrimaryKey = false, Comment = "生産者コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 891, Length = 72, Name = "seisansha_name_hojinkaku_nashi", IsPrimaryKey = false, Comment = "生産者名(法人格無)", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 963, Length = 20, Name = "sanchi_name", IsPrimaryKey = false, Comment = "産地名", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 983, Length = 6, Name = "banushi_code", IsPrimaryKey = false, Comment = "馬主コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 989, Length = 64, Name = "banushi_name_hojinkaku_nashi", IsPrimaryKey = false, Comment = "馬主名(法人格無)", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1053, Length = 9, Name = "heichi_hon_shokin_ruikei", IsPrimaryKey = false, Comment = "平地本賞金累計", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1062, Length = 9, Name = "shogai_hon_shokin_ruikei", IsPrimaryKey = false, Comment = "障害本賞金累計", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1071, Length = 9, Name = "heichi_fuka_shokin_ruikei", IsPrimaryKey = false, Comment = "平地付加賞金累計", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1080, Length = 9, Name = "shogai_fuka_shokin_ruikei", IsPrimaryKey = false, Comment = "障害付加賞金累計", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1089, Length = 9, Name = "heichi_shutoku_shokin_ruikei", IsPrimaryKey = false, Comment = "平地収得賞金累計", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1098, Length = 9, Name = "shogai_shutoku_shokin_ruikei", IsPrimaryKey = false, Comment = "障害収得賞金累計", DataType = "CHAR" },
                        
                        // 総合着回数 (6回繰り返し)
                        new NormalFieldDefinition { Position = 1107, Length = 3, Name = "sogo_chaku_kaisu_1", IsPrimaryKey = false, Comment = "総合着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1110, Length = 3, Name = "sogo_chaku_kaisu_2", IsPrimaryKey = false, Comment = "総合着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1113, Length = 3, Name = "sogo_chaku_kaisu_3", IsPrimaryKey = false, Comment = "総合着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1116, Length = 3, Name = "sogo_chaku_kaisu_4", IsPrimaryKey = false, Comment = "総合着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1119, Length = 3, Name = "sogo_chaku_kaisu_5", IsPrimaryKey = false, Comment = "総合着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1122, Length = 3, Name = "sogo_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "総合着回数着外", DataType = "CHAR" },
                        
                        // 中央合計着回数 (6回繰り返し)
                        new NormalFieldDefinition { Position = 1125, Length = 3, Name = "chuo_gokei_chaku_kaisu_1", IsPrimaryKey = false, Comment = "中央合計着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1128, Length = 3, Name = "chuo_gokei_chaku_kaisu_2", IsPrimaryKey = false, Comment = "中央合計着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1131, Length = 3, Name = "chuo_gokei_chaku_kaisu_3", IsPrimaryKey = false, Comment = "中央合計着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1134, Length = 3, Name = "chuo_gokei_chaku_kaisu_4", IsPrimaryKey = false, Comment = "中央合計着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1137, Length = 3, Name = "chuo_gokei_chaku_kaisu_5", IsPrimaryKey = false, Comment = "中央合計着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1140, Length = 3, Name = "chuo_gokei_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "中央合計着回数着外", DataType = "CHAR" },
                        
                        // 芝直・着回数 (6回繰り返し)
                        new NormalFieldDefinition { Position = 1143, Length = 3, Name = "shiba_choku_chaku_kaisu_1", IsPrimaryKey = false, Comment = "芝直着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1146, Length = 3, Name = "shiba_choku_chaku_kaisu_2", IsPrimaryKey = false, Comment = "芝直着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1149, Length = 3, Name = "shiba_choku_chaku_kaisu_3", IsPrimaryKey = false, Comment = "芝直着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1152, Length = 3, Name = "shiba_choku_chaku_kaisu_4", IsPrimaryKey = false, Comment = "芝直着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1155, Length = 3, Name = "shiba_choku_chaku_kaisu_5", IsPrimaryKey = false, Comment = "芝直着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1158, Length = 3, Name = "shiba_choku_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "芝直着回数着外", DataType = "CHAR" },
                        
                        // 芝右・着回数 (6回繰り返し)
                        new NormalFieldDefinition { Position = 1161, Length = 3, Name = "shiba_migi_chaku_kaisu_1", IsPrimaryKey = false, Comment = "芝右着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1164, Length = 3, Name = "shiba_migi_chaku_kaisu_2", IsPrimaryKey = false, Comment = "芝右着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1167, Length = 3, Name = "shiba_migi_chaku_kaisu_3", IsPrimaryKey = false, Comment = "芝右着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1170, Length = 3, Name = "shiba_migi_chaku_kaisu_4", IsPrimaryKey = false, Comment = "芝右着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1173, Length = 3, Name = "shiba_migi_chaku_kaisu_5", IsPrimaryKey = false, Comment = "芝右着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1176, Length = 3, Name = "shiba_migi_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "芝右着回数着外", DataType = "CHAR" },
                        
                        // 芝左・着回数 (6回繰り返し)
                        new NormalFieldDefinition { Position = 1179, Length = 3, Name = "shiba_hidari_chaku_kaisu_1", IsPrimaryKey = false, Comment = "芝左着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1182, Length = 3, Name = "shiba_hidari_chaku_kaisu_2", IsPrimaryKey = false, Comment = "芝左着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1185, Length = 3, Name = "shiba_hidari_chaku_kaisu_3", IsPrimaryKey = false, Comment = "芝左着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1188, Length = 3, Name = "shiba_hidari_chaku_kaisu_4", IsPrimaryKey = false, Comment = "芝左着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1191, Length = 3, Name = "shiba_hidari_chaku_kaisu_5", IsPrimaryKey = false, Comment = "芝左着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1194, Length = 3, Name = "shiba_hidari_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "芝左着回数着外", DataType = "CHAR" },
                        
                        // ダ直・着回数 (6回繰り返し)
                        new NormalFieldDefinition { Position = 1197, Length = 3, Name = "dirt_choku_chaku_kaisu_1", IsPrimaryKey = false, Comment = "ダ直着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1200, Length = 3, Name = "dirt_choku_chaku_kaisu_2", IsPrimaryKey = false, Comment = "ダ直着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1203, Length = 3, Name = "dirt_choku_chaku_kaisu_3", IsPrimaryKey = false, Comment = "ダ直着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1206, Length = 3, Name = "dirt_choku_chaku_kaisu_4", IsPrimaryKey = false, Comment = "ダ直着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1209, Length = 3, Name = "dirt_choku_chaku_kaisu_5", IsPrimaryKey = false, Comment = "ダ直着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1212, Length = 3, Name = "dirt_choku_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "ダ直着回数着外", DataType = "CHAR" },
                        
                        // ダ右・着回数 (6回繰り返し)
                        new NormalFieldDefinition { Position = 1215, Length = 3, Name = "dirt_migi_chaku_kaisu_1", IsPrimaryKey = false, Comment = "ダ右着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1218, Length = 3, Name = "dirt_migi_chaku_kaisu_2", IsPrimaryKey = false, Comment = "ダ右着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1221, Length = 3, Name = "dirt_migi_chaku_kaisu_3", IsPrimaryKey = false, Comment = "ダ右着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1224, Length = 3, Name = "dirt_migi_chaku_kaisu_4", IsPrimaryKey = false, Comment = "ダ右着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1227, Length = 3, Name = "dirt_migi_chaku_kaisu_5", IsPrimaryKey = false, Comment = "ダ右着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1230, Length = 3, Name = "dirt_migi_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "ダ右着回数着外", DataType = "CHAR" },
                        
                        // ダ左・着回数 (6回繰り返し)
                        new NormalFieldDefinition { Position = 1233, Length = 3, Name = "dirt_hidari_chaku_kaisu_1", IsPrimaryKey = false, Comment = "ダ左着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1236, Length = 3, Name = "dirt_hidari_chaku_kaisu_2", IsPrimaryKey = false, Comment = "ダ左着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1239, Length = 3, Name = "dirt_hidari_chaku_kaisu_3", IsPrimaryKey = false, Comment = "ダ左着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1242, Length = 3, Name = "dirt_hidari_chaku_kaisu_4", IsPrimaryKey = false, Comment = "ダ左着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1245, Length = 3, Name = "dirt_hidari_chaku_kaisu_5", IsPrimaryKey = false, Comment = "ダ左着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1248, Length = 3, Name = "dirt_hidari_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "ダ左着回数着外", DataType = "CHAR" },
                        
                        // 障害・着回数 (6回繰り返し)
                        new NormalFieldDefinition { Position = 1251, Length = 3, Name = "shogai_chaku_kaisu_1", IsPrimaryKey = false, Comment = "障害着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1254, Length = 3, Name = "shogai_chaku_kaisu_2", IsPrimaryKey = false, Comment = "障害着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1257, Length = 3, Name = "shogai_chaku_kaisu_3", IsPrimaryKey = false, Comment = "障害着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1260, Length = 3, Name = "shogai_chaku_kaisu_4", IsPrimaryKey = false, Comment = "障害着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1263, Length = 3, Name = "shogai_chaku_kaisu_5", IsPrimaryKey = false, Comment = "障害着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1266, Length = 3, Name = "shogai_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "障害着回数着外", DataType = "CHAR" },
                        
                        // 芝良・着回数 (6回繰り返し)
                        new NormalFieldDefinition { Position = 1269, Length = 3, Name = "shiba_ryo_chaku_kaisu_1", IsPrimaryKey = false, Comment = "芝良着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1272, Length = 3, Name = "shiba_ryo_chaku_kaisu_2", IsPrimaryKey = false, Comment = "芝良着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1275, Length = 3, Name = "shiba_ryo_chaku_kaisu_3", IsPrimaryKey = false, Comment = "芝良着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1278, Length = 3, Name = "shiba_ryo_chaku_kaisu_4", IsPrimaryKey = false, Comment = "芝良着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1281, Length = 3, Name = "shiba_ryo_chaku_kaisu_5", IsPrimaryKey = false, Comment = "芝良着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1284, Length = 3, Name = "shiba_ryo_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "芝良着回数着外", DataType = "CHAR" },
                        
                        // 芝稍・着回数 (6回繰り返し)
                        new NormalFieldDefinition { Position = 1287, Length = 3, Name = "shiba_yayaomo_chaku_kaisu_1", IsPrimaryKey = false, Comment = "芝稍着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1290, Length = 3, Name = "shiba_yayaomo_chaku_kaisu_2", IsPrimaryKey = false, Comment = "芝稍着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1293, Length = 3, Name = "shiba_yayaomo_chaku_kaisu_3", IsPrimaryKey = false, Comment = "芝稍着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1296, Length = 3, Name = "shiba_yayaomo_chaku_kaisu_4", IsPrimaryKey = false, Comment = "芝稍着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1299, Length = 3, Name = "shiba_yayaomo_chaku_kaisu_5", IsPrimaryKey = false, Comment = "芝稍着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1302, Length = 3, Name = "shiba_yayaomo_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "芝稍着回数着外", DataType = "CHAR" },
                        
                        // 芝重・着回数 (6回繰り返し)
                        new NormalFieldDefinition { Position = 1305, Length = 3, Name = "shiba_omo_chaku_kaisu_1", IsPrimaryKey = false, Comment = "芝重着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1308, Length = 3, Name = "shiba_omo_chaku_kaisu_2", IsPrimaryKey = false, Comment = "芝重着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1311, Length = 3, Name = "shiba_omo_chaku_kaisu_3", IsPrimaryKey = false, Comment = "芝重着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1314, Length = 3, Name = "shiba_omo_chaku_kaisu_4", IsPrimaryKey = false, Comment = "芝重着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1317, Length = 3, Name = "shiba_omo_chaku_kaisu_5", IsPrimaryKey = false, Comment = "芝重着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1320, Length = 3, Name = "shiba_omo_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "芝重着回数着外", DataType = "CHAR" },
                        
                        // 芝不・着回数 (6回繰り返し)
                        new NormalFieldDefinition { Position = 1323, Length = 3, Name = "shiba_furyo_chaku_kaisu_1", IsPrimaryKey = false, Comment = "芝不着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1326, Length = 3, Name = "shiba_furyo_chaku_kaisu_2", IsPrimaryKey = false, Comment = "芝不着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1329, Length = 3, Name = "shiba_furyo_chaku_kaisu_3", IsPrimaryKey = false, Comment = "芝不着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1332, Length = 3, Name = "shiba_furyo_chaku_kaisu_4", IsPrimaryKey = false, Comment = "芝不着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1335, Length = 3, Name = "shiba_furyo_chaku_kaisu_5", IsPrimaryKey = false, Comment = "芝不着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1338, Length = 3, Name = "shiba_furyo_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "芝不着回数着外", DataType = "CHAR" },
                        
                        // ダ良・着回数 (6回繰り返し)
                        new NormalFieldDefinition { Position = 1341, Length = 3, Name = "dirt_ryo_chaku_kaisu_1", IsPrimaryKey = false, Comment = "ダ良着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1344, Length = 3, Name = "dirt_ryo_chaku_kaisu_2", IsPrimaryKey = false, Comment = "ダ良着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1347, Length = 3, Name = "dirt_ryo_chaku_kaisu_3", IsPrimaryKey = false, Comment = "ダ良着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1350, Length = 3, Name = "dirt_ryo_chaku_kaisu_4", IsPrimaryKey = false, Comment = "ダ良着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1353, Length = 3, Name = "dirt_ryo_chaku_kaisu_5", IsPrimaryKey = false, Comment = "ダ良着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1356, Length = 3, Name = "dirt_ryo_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "ダ良着回数着外", DataType = "CHAR" },
                        
                        // ダ稍・着回数 (6回繰り返し)
                        new NormalFieldDefinition { Position = 1359, Length = 3, Name = "dirt_yayaomo_chaku_kaisu_1", IsPrimaryKey = false, Comment = "ダ稍着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1362, Length = 3, Name = "dirt_yayaomo_chaku_kaisu_2", IsPrimaryKey = false, Comment = "ダ稍着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1365, Length = 3, Name = "dirt_yayaomo_chaku_kaisu_3", IsPrimaryKey = false, Comment = "ダ稍着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1368, Length = 3, Name = "dirt_yayaomo_chaku_kaisu_4", IsPrimaryKey = false, Comment = "ダ稍着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1371, Length = 3, Name = "dirt_yayaomo_chaku_kaisu_5", IsPrimaryKey = false, Comment = "ダ稍着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1374, Length = 3, Name = "dirt_yayaomo_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "ダ稍着回数着外", DataType = "CHAR" },
                        
                        // ダ重・着回数 (6回繰り返し)
                        new NormalFieldDefinition { Position = 1377, Length = 3, Name = "dirt_omo_chaku_kaisu_1", IsPrimaryKey = false, Comment = "ダ重着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1380, Length = 3, Name = "dirt_omo_chaku_kaisu_2", IsPrimaryKey = false, Comment = "ダ重着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1383, Length = 3, Name = "dirt_omo_chaku_kaisu_3", IsPrimaryKey = false, Comment = "ダ重着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1386, Length = 3, Name = "dirt_omo_chaku_kaisu_4", IsPrimaryKey = false, Comment = "ダ重着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1389, Length = 3, Name = "dirt_omo_chaku_kaisu_5", IsPrimaryKey = false, Comment = "ダ重着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1392, Length = 3, Name = "dirt_omo_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "ダ重着回数着外", DataType = "CHAR" },
                        
                        // ダ不・着回数 (6回繰り返し)
                        new NormalFieldDefinition { Position = 1395, Length = 3, Name = "dirt_furyo_chaku_kaisu_1", IsPrimaryKey = false, Comment = "ダ不着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1398, Length = 3, Name = "dirt_furyo_chaku_kaisu_2", IsPrimaryKey = false, Comment = "ダ不着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1401, Length = 3, Name = "dirt_furyo_chaku_kaisu_3", IsPrimaryKey = false, Comment = "ダ不着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1404, Length = 3, Name = "dirt_furyo_chaku_kaisu_4", IsPrimaryKey = false, Comment = "ダ不着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1407, Length = 3, Name = "dirt_furyo_chaku_kaisu_5", IsPrimaryKey = false, Comment = "ダ不着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1410, Length = 3, Name = "dirt_furyo_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "ダ不着回数着外", DataType = "CHAR" },
                        
                        // 障良・着回数 (6回繰り返し)
                        new NormalFieldDefinition { Position = 1413, Length = 3, Name = "shogai_ryo_chaku_kaisu_1", IsPrimaryKey = false, Comment = "障良着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1416, Length = 3, Name = "shogai_ryo_chaku_kaisu_2", IsPrimaryKey = false, Comment = "障良着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1419, Length = 3, Name = "shogai_ryo_chaku_kaisu_3", IsPrimaryKey = false, Comment = "障良着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1422, Length = 3, Name = "shogai_ryo_chaku_kaisu_4", IsPrimaryKey = false, Comment = "障良着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1425, Length = 3, Name = "shogai_ryo_chaku_kaisu_5", IsPrimaryKey = false, Comment = "障良着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1428, Length = 3, Name = "shogai_ryo_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "障良着回数着外", DataType = "CHAR" },
                        
                        // 障稍・着回数 (6回繰り返し)
                        new NormalFieldDefinition { Position = 1431, Length = 3, Name = "shogai_yayaomo_chaku_kaisu_1", IsPrimaryKey = false, Comment = "障稍着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1434, Length = 3, Name = "shogai_yayaomo_chaku_kaisu_2", IsPrimaryKey = false, Comment = "障稍着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1437, Length = 3, Name = "shogai_yayaomo_chaku_kaisu_3", IsPrimaryKey = false, Comment = "障稍着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1440, Length = 3, Name = "shogai_yayaomo_chaku_kaisu_4", IsPrimaryKey = false, Comment = "障稍着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1443, Length = 3, Name = "shogai_yayaomo_chaku_kaisu_5", IsPrimaryKey = false, Comment = "障稍着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1446, Length = 3, Name = "shogai_yayaomo_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "障稍着回数着外", DataType = "CHAR" },
                        
                        // 障重・着回数 (6回繰り返し)
                        new NormalFieldDefinition { Position = 1449, Length = 3, Name = "shogai_omo_chaku_kaisu_1", IsPrimaryKey = false, Comment = "障重着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1452, Length = 3, Name = "shogai_omo_chaku_kaisu_2", IsPrimaryKey = false, Comment = "障重着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1455, Length = 3, Name = "shogai_omo_chaku_kaisu_3", IsPrimaryKey = false, Comment = "障重着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1458, Length = 3, Name = "shogai_omo_chaku_kaisu_4", IsPrimaryKey = false, Comment = "障重着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1461, Length = 3, Name = "shogai_omo_chaku_kaisu_5", IsPrimaryKey = false, Comment = "障重着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1464, Length = 3, Name = "shogai_omo_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "障重着回数着外", DataType = "CHAR" },
                        
                        // 障不・着回数 (6回繰り返し)
                        new NormalFieldDefinition { Position = 1467, Length = 3, Name = "shogai_furyo_chaku_kaisu_1", IsPrimaryKey = false, Comment = "障不着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1470, Length = 3, Name = "shogai_furyo_chaku_kaisu_2", IsPrimaryKey = false, Comment = "障不着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1473, Length = 3, Name = "shogai_furyo_chaku_kaisu_3", IsPrimaryKey = false, Comment = "障不着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1476, Length = 3, Name = "shogai_furyo_chaku_kaisu_4", IsPrimaryKey = false, Comment = "障不着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1479, Length = 3, Name = "shogai_furyo_chaku_kaisu_5", IsPrimaryKey = false, Comment = "障不着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1482, Length = 3, Name = "shogai_furyo_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "障不着回数着外", DataType = "CHAR" },
                        
                        // 芝1600以下・着回数 (6回繰り返し)
                        new NormalFieldDefinition { Position = 1485, Length = 3, Name = "shiba_1600_ika_chaku_kaisu_1", IsPrimaryKey = false, Comment = "芝1600以下着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1488, Length = 3, Name = "shiba_1600_ika_chaku_kaisu_2", IsPrimaryKey = false, Comment = "芝1600以下着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1491, Length = 3, Name = "shiba_1600_ika_chaku_kaisu_3", IsPrimaryKey = false, Comment = "芝1600以下着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1494, Length = 3, Name = "shiba_1600_ika_chaku_kaisu_4", IsPrimaryKey = false, Comment = "芝1600以下着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1497, Length = 3, Name = "shiba_1600_ika_chaku_kaisu_5", IsPrimaryKey = false, Comment = "芝1600以下着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1500, Length = 3, Name = "shiba_1600_ika_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "芝1600以下着回数着外", DataType = "CHAR" },
                        
                        // 芝2200以下・着回数 (6回繰り返し)
                        new NormalFieldDefinition { Position = 1503, Length = 3, Name = "shiba_1601_2200_chaku_kaisu_1", IsPrimaryKey = false, Comment = "芝2200以下着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1506, Length = 3, Name = "shiba_1601_2200_chaku_kaisu_2", IsPrimaryKey = false, Comment = "芝2200以下着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1509, Length = 3, Name = "shiba_1601_2200_chaku_kaisu_3", IsPrimaryKey = false, Comment = "芝2200以下着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1512, Length = 3, Name = "shiba_1601_2200_chaku_kaisu_4", IsPrimaryKey = false, Comment = "芝2200以下着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1515, Length = 3, Name = "shiba_1601_2200_chaku_kaisu_5", IsPrimaryKey = false, Comment = "芝2200以下着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1518, Length = 3, Name = "shiba_1601_2200_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "芝2200以下着回数着外", DataType = "CHAR" },
                        
                        // 芝2200超・着回数 (6回繰り返し)
                        new NormalFieldDefinition { Position = 1521, Length = 3, Name = "shiba_2201_ijo_chaku_kaisu_1", IsPrimaryKey = false, Comment = "芝2200超着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1524, Length = 3, Name = "shiba_2201_ijo_chaku_kaisu_2", IsPrimaryKey = false, Comment = "芝2200超着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1527, Length = 3, Name = "shiba_2201_ijo_chaku_kaisu_3", IsPrimaryKey = false, Comment = "芝2200超着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1530, Length = 3, Name = "shiba_2201_ijo_chaku_kaisu_4", IsPrimaryKey = false, Comment = "芝2200超着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1533, Length = 3, Name = "shiba_2201_ijo_chaku_kaisu_5", IsPrimaryKey = false, Comment = "芝2200超着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1536, Length = 3, Name = "shiba_2201_ijo_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "芝2200超着回数着外", DataType = "CHAR" },
                        
                        // ダ1600以下・着回数 (6回繰り返し)
                        new NormalFieldDefinition { Position = 1539, Length = 3, Name = "dirt_1600_ika_chaku_kaisu_1", IsPrimaryKey = false, Comment = "ダ1600以下着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1542, Length = 3, Name = "dirt_1600_ika_chaku_kaisu_2", IsPrimaryKey = false, Comment = "ダ1600以下着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1545, Length = 3, Name = "dirt_1600_ika_chaku_kaisu_3", IsPrimaryKey = false, Comment = "ダ1600以下着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1548, Length = 3, Name = "dirt_1600_ika_chaku_kaisu_4", IsPrimaryKey = false, Comment = "ダ1600以下着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1551, Length = 3, Name = "dirt_1600_ika_chaku_kaisu_5", IsPrimaryKey = false, Comment = "ダ1600以下着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1554, Length = 3, Name = "dirt_1600_ika_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "ダ1600以下着回数着外", DataType = "CHAR" },
                        
                        // ダ2200以下・着回数 (6回繰り返し)
                        new NormalFieldDefinition { Position = 1557, Length = 3, Name = "dirt_1601_2200_chaku_kaisu_1", IsPrimaryKey = false, Comment = "ダ2200以下着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1560, Length = 3, Name = "dirt_1601_2200_chaku_kaisu_2", IsPrimaryKey = false, Comment = "ダ2200以下着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1563, Length = 3, Name = "dirt_1601_2200_chaku_kaisu_3", IsPrimaryKey = false, Comment = "ダ2200以下着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1566, Length = 3, Name = "dirt_1601_2200_chaku_kaisu_4", IsPrimaryKey = false, Comment = "ダ2200以下着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1569, Length = 3, Name = "dirt_1601_2200_chaku_kaisu_5", IsPrimaryKey = false, Comment = "ダ2200以下着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1572, Length = 3, Name = "dirt_1601_2200_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "ダ2200以下着回数着外", DataType = "CHAR" },
                        
                        // ダ2200超・着回数 (6回繰り返し)
                        new NormalFieldDefinition { Position = 1575, Length = 3, Name = "dirt_2201_ijo_chaku_kaisu_1", IsPrimaryKey = false, Comment = "ダ2200超着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1578, Length = 3, Name = "dirt_2201_ijo_chaku_kaisu_2", IsPrimaryKey = false, Comment = "ダ2200超着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1581, Length = 3, Name = "dirt_2201_ijo_chaku_kaisu_3", IsPrimaryKey = false, Comment = "ダ2200超着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1584, Length = 3, Name = "dirt_2201_ijo_chaku_kaisu_4", IsPrimaryKey = false, Comment = "ダ2200超着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1587, Length = 3, Name = "dirt_2201_ijo_chaku_kaisu_5", IsPrimaryKey = false, Comment = "ダ2200超着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1590, Length = 3, Name = "dirt_2201_ijo_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "ダ2200超着回数着外", DataType = "CHAR" },
                        
                        // 脚質傾向 (4回繰り返し)
                        new NormalFieldDefinition { Position = 1593, Length = 3, Name = "kyakushitsu_keiko_nige", IsPrimaryKey = false, Comment = "脚質傾向逃げ", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1596, Length = 3, Name = "kyakushitsu_keiko_senko", IsPrimaryKey = false, Comment = "脚質傾向先行", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1599, Length = 3, Name = "kyakushitsu_keiko_sashi", IsPrimaryKey = false, Comment = "脚質傾向差し", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1602, Length = 3, Name = "kyakushitsu_keiko_oikomi", IsPrimaryKey = false, Comment = "脚質傾向追込", DataType = "CHAR" },
                        
                        new NormalFieldDefinition { Position = 1605, Length = 3, Name = "toroku_race_su", IsPrimaryKey = false, Comment = "登録レース数", DataType = "CHAR" }
                    }
                }
            };
            recordDefinitions.Add(umRecord);

            // KS - 騎手マスタ
            var ksRecord = new RecordDefinition
            {
                RecordTypeId = "KS",
                CreationDateField = new FieldDefinition { Position = 4, Length = 8 },
                Table = new TableDefinition
                {
                    Name = "kishu_master",
                    Comment = "騎手マスタ",
                    Fields = new List<FieldDefinition>
                    {
                        new NormalFieldDefinition { Position = 1, Length = 2, Name = "record_type_id", IsPrimaryKey = false, Comment = "レコード種別ID", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 3, Length = 1, Name = "data_kubun", IsPrimaryKey = false, Comment = "データ区分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 12, Length = 5, Name = "kishu_code", IsPrimaryKey = true, Comment = "騎手コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 17, Length = 1, Name = "kishu_massho_kubun", IsPrimaryKey = false, Comment = "騎手抹消区分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 18, Length = 8, Name = "kishu_menkyo_kofu_date", IsPrimaryKey = false, Comment = "騎手免許交付年月日", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 26, Length = 8, Name = "kishu_menkyo_massho_date", IsPrimaryKey = false, Comment = "騎手免許抹消年月日", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 34, Length = 8, Name = "birth_date", IsPrimaryKey = false, Comment = "生年月日", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 42, Length = 34, Name = "kishu_name", IsPrimaryKey = false, Comment = "騎手名", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 76, Length = 34, Name = "yobi", IsPrimaryKey = false, Comment = "予備", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 110, Length = 30, Name = "kishu_name_hankaku_kana", IsPrimaryKey = false, Comment = "騎手名半角カナ", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 140, Length = 8, Name = "kishu_name_short", IsPrimaryKey = false, Comment = "騎手名略称", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 148, Length = 80, Name = "kishu_name_eng", IsPrimaryKey = false, Comment = "騎手名欧字", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 228, Length = 1, Name = "seibetsu_kubun", IsPrimaryKey = false, Comment = "性別区分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 229, Length = 1, Name = "kijo_shikaku_code", IsPrimaryKey = false, Comment = "騎乗資格コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 230, Length = 1, Name = "kishu_minarai_code", IsPrimaryKey = false, Comment = "騎手見習コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 231, Length = 1, Name = "kishu_tozai_shozoku_code", IsPrimaryKey = false, Comment = "騎手東西所属コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 232, Length = 20, Name = "shotai_chiki_name", IsPrimaryKey = false, Comment = "招待地域名", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 252, Length = 5, Name = "shozoku_chokyoshi_code", IsPrimaryKey = false, Comment = "所属調教師コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 257, Length = 8, Name = "shozoku_chokyoshi_name_short", IsPrimaryKey = false, Comment = "所属調教師名略称", DataType = "CHAR" },
                        
                        // 初騎乗情報 (平地・障害の2回繰り返し)
                        new RepeatFieldDefinition
                        {
                            Position = 265,
                            Length = 67,
                            RepeatCount = 2,
                            Table = new TableDefinition
                            {
                                Name = "hatsu_kijo_joho",
                                Comment = "初騎乗情報",
                                Fields = new List<FieldDefinition>
                                {
                                    new NormalFieldDefinition { Position = 1, Length = 16, Name = "race_id", IsPrimaryKey = false, Comment = "年月日場回日R", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 17, Length = 2, Name = "shusso_tosu", IsPrimaryKey = false, Comment = "出走頭数", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 19, Length = 10, Name = "ketto_toroku_number", IsPrimaryKey = false, Comment = "血統登録番号", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 29, Length = 36, Name = "uma_name", IsPrimaryKey = false, Comment = "馬名", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 65, Length = 2, Name = "kakutei_chakujun", IsPrimaryKey = false, Comment = "確定着順", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 67, Length = 1, Name = "ijo_kubun_code", IsPrimaryKey = false, Comment = "異常区分コード", DataType = "CHAR" }
                                }
                            }
                        },
                        
                        // 初勝利情報 (平地・障害の2回繰り返し)
                        new RepeatFieldDefinition
                        {
                            Position = 399,
                            Length = 64,
                            RepeatCount = 2,
                            Table = new TableDefinition
                            {
                                Name = "hatsu_shori_joho",
                                Comment = "初勝利情報",
                                Fields = new List<FieldDefinition>
                                {
                                    new NormalFieldDefinition { Position = 1, Length = 16, Name = "race_id", IsPrimaryKey = false, Comment = "年月日場回日R", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 17, Length = 2, Name = "shusso_tosu", IsPrimaryKey = false, Comment = "出走頭数", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 19, Length = 10, Name = "ketto_toroku_number", IsPrimaryKey = false, Comment = "血統登録番号", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 29, Length = 36, Name = "uma_name", IsPrimaryKey = false, Comment = "馬名", DataType = "CHAR" }
                                }
                            }
                        },
                        
                        // 最近重賞勝利情報 (3回繰り返し)
                        new RepeatFieldDefinition
                        {
                            Position = 527,
                            Length = 163,
                            RepeatCount = 3,
                            Table = new TableDefinition
                            {
                                Name = "saikin_jusho_shori_joho",
                                Comment = "最近重賞勝利情報",
                                Fields = new List<FieldDefinition>
                                {
                                    new NormalFieldDefinition { Position = 1, Length = 16, Name = "race_id", IsPrimaryKey = false, Comment = "年月日場回日R", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 17, Length = 60, Name = "kyoso_name_hondai", IsPrimaryKey = false, Comment = "競走名本題", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 77, Length = 20, Name = "kyoso_name_short_10", IsPrimaryKey = false, Comment = "競走名略称10文字", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 97, Length = 12, Name = "kyoso_name_short_6", IsPrimaryKey = false, Comment = "競走名略称6文字", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 109, Length = 6, Name = "kyoso_name_short_3", IsPrimaryKey = false, Comment = "競走名略称3文字", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 115, Length = 1, Name = "grade_code", IsPrimaryKey = false, Comment = "グレードコード", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 116, Length = 2, Name = "shusso_tosu", IsPrimaryKey = false, Comment = "出走頭数", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 118, Length = 10, Name = "ketto_toroku_number", IsPrimaryKey = false, Comment = "血統登録番号", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 128, Length = 36, Name = "uma_name", IsPrimaryKey = false, Comment = "馬名", DataType = "CHAR" }
                                }
                            }
                        },
                        
                        // 本年・前年・累計成績情報 (3回繰り返し)
                        new RepeatFieldDefinition
                        {
                            Position = 1016,
                            Length = 1052,
                            RepeatCount = 3,
                            Table = new TableDefinition
                            {
                                Name = "seiseki_joho",
                                Comment = "本年・前年・累計成績情報",
                                Fields = new List<FieldDefinition>
                                {
                                    new NormalFieldDefinition { Position = 1, Length = 4, Name = "settei_year", IsPrimaryKey = false, Comment = "設定年", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 5, Length = 10, Name = "heichi_hon_shokin_gokei", IsPrimaryKey = false, Comment = "平地本賞金合計", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 15, Length = 10, Name = "shogai_hon_shokin_gokei", IsPrimaryKey = false, Comment = "障害本賞金合計", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 25, Length = 10, Name = "heichi_fuka_shokin_gokei", IsPrimaryKey = false, Comment = "平地付加賞金合計", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 35, Length = 10, Name = "shogai_fuka_shokin_gokei", IsPrimaryKey = false, Comment = "障害付加賞金合計", DataType = "CHAR" },
                                    
                                    // 平地着回数 (1着～5着及び着外の6回)
                                    new NormalFieldDefinition { Position = 45, Length = 6, Name = "heichi_chaku_kaisu_1", IsPrimaryKey = false, Comment = "平地着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 51, Length = 6, Name = "heichi_chaku_kaisu_2", IsPrimaryKey = false, Comment = "平地着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 57, Length = 6, Name = "heichi_chaku_kaisu_3", IsPrimaryKey = false, Comment = "平地着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 63, Length = 6, Name = "heichi_chaku_kaisu_4", IsPrimaryKey = false, Comment = "平地着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 69, Length = 6, Name = "heichi_chaku_kaisu_5", IsPrimaryKey = false, Comment = "平地着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 75, Length = 6, Name = "heichi_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "平地着回数着外", DataType = "CHAR" },
                                    
                                    // 障害着回数 (1着～5着及び着外の6回)
                                    new NormalFieldDefinition { Position = 81, Length = 6, Name = "shogai_chaku_kaisu_1", IsPrimaryKey = false, Comment = "障害着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 87, Length = 6, Name = "shogai_chaku_kaisu_2", IsPrimaryKey = false, Comment = "障害着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 93, Length = 6, Name = "shogai_chaku_kaisu_3", IsPrimaryKey = false, Comment = "障害着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 99, Length = 6, Name = "shogai_chaku_kaisu_4", IsPrimaryKey = false, Comment = "障害着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 105, Length = 6, Name = "shogai_chaku_kaisu_5", IsPrimaryKey = false, Comment = "障害着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 111, Length = 6, Name = "shogai_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "障害着回数着外", DataType = "CHAR" },
                                    
                                    // 競馬場別着回数 (札幌)
                                    new NormalFieldDefinition { Position = 117, Length = 6, Name = "sapporo_heichi_chaku_kaisu_1", IsPrimaryKey = false, Comment = "札幌平地着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 123, Length = 6, Name = "sapporo_heichi_chaku_kaisu_2", IsPrimaryKey = false, Comment = "札幌平地着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 129, Length = 6, Name = "sapporo_heichi_chaku_kaisu_3", IsPrimaryKey = false, Comment = "札幌平地着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 135, Length = 6, Name = "sapporo_heichi_chaku_kaisu_4", IsPrimaryKey = false, Comment = "札幌平地着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 141, Length = 6, Name = "sapporo_heichi_chaku_kaisu_5", IsPrimaryKey = false, Comment = "札幌平地着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 147, Length = 6, Name = "sapporo_heichi_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "札幌平地着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 153, Length = 6, Name = "sapporo_shogai_chaku_kaisu_1", IsPrimaryKey = false, Comment = "札幌障害着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 159, Length = 6, Name = "sapporo_shogai_chaku_kaisu_2", IsPrimaryKey = false, Comment = "札幌障害着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 165, Length = 6, Name = "sapporo_shogai_chaku_kaisu_3", IsPrimaryKey = false, Comment = "札幌障害着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 171, Length = 6, Name = "sapporo_shogai_chaku_kaisu_4", IsPrimaryKey = false, Comment = "札幌障害着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 177, Length = 6, Name = "sapporo_shogai_chaku_kaisu_5", IsPrimaryKey = false, Comment = "札幌障害着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 183, Length = 6, Name = "sapporo_shogai_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "札幌障害着回数着外", DataType = "CHAR" },
                                    
                                    // 競馬場別着回数 (函館)
                                    new NormalFieldDefinition { Position = 189, Length = 6, Name = "hakodate_heichi_chaku_kaisu_1", IsPrimaryKey = false, Comment = "函館平地着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 195, Length = 6, Name = "hakodate_heichi_chaku_kaisu_2", IsPrimaryKey = false, Comment = "函館平地着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 201, Length = 6, Name = "hakodate_heichi_chaku_kaisu_3", IsPrimaryKey = false, Comment = "函館平地着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 207, Length = 6, Name = "hakodate_heichi_chaku_kaisu_4", IsPrimaryKey = false, Comment = "函館平地着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 213, Length = 6, Name = "hakodate_heichi_chaku_kaisu_5", IsPrimaryKey = false, Comment = "函館平地着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 219, Length = 6, Name = "hakodate_heichi_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "函館平地着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 225, Length = 6, Name = "hakodate_shogai_chaku_kaisu_1", IsPrimaryKey = false, Comment = "函館障害着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 231, Length = 6, Name = "hakodate_shogai_chaku_kaisu_2", IsPrimaryKey = false, Comment = "函館障害着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 237, Length = 6, Name = "hakodate_shogai_chaku_kaisu_3", IsPrimaryKey = false, Comment = "函館障害着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 243, Length = 6, Name = "hakodate_shogai_chaku_kaisu_4", IsPrimaryKey = false, Comment = "函館障害着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 249, Length = 6, Name = "hakodate_shogai_chaku_kaisu_5", IsPrimaryKey = false, Comment = "函館障害着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 255, Length = 6, Name = "hakodate_shogai_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "函館障害着回数着外", DataType = "CHAR" },
                                    
                                    // 競馬場別着回数 (福島)
                                    new NormalFieldDefinition { Position = 261, Length = 6, Name = "fukushima_heichi_chaku_kaisu_1", IsPrimaryKey = false, Comment = "福島平地着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 267, Length = 6, Name = "fukushima_heichi_chaku_kaisu_2", IsPrimaryKey = false, Comment = "福島平地着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 273, Length = 6, Name = "fukushima_heichi_chaku_kaisu_3", IsPrimaryKey = false, Comment = "福島平地着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 279, Length = 6, Name = "fukushima_heichi_chaku_kaisu_4", IsPrimaryKey = false, Comment = "福島平地着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 285, Length = 6, Name = "fukushima_heichi_chaku_kaisu_5", IsPrimaryKey = false, Comment = "福島平地着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 291, Length = 6, Name = "fukushima_heichi_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "福島平地着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 297, Length = 6, Name = "fukushima_shogai_chaku_kaisu_1", IsPrimaryKey = false, Comment = "福島障害着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 303, Length = 6, Name = "fukushima_shogai_chaku_kaisu_2", IsPrimaryKey = false, Comment = "福島障害着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 309, Length = 6, Name = "fukushima_shogai_chaku_kaisu_3", IsPrimaryKey = false, Comment = "福島障害着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 315, Length = 6, Name = "fukushima_shogai_chaku_kaisu_4", IsPrimaryKey = false, Comment = "福島障害着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 321, Length = 6, Name = "fukushima_shogai_chaku_kaisu_5", IsPrimaryKey = false, Comment = "福島障害着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 327, Length = 6, Name = "fukushima_shogai_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "福島障害着回数着外", DataType = "CHAR" },
                                    
                                    // 競馬場別着回数 (新潟)
                                    new NormalFieldDefinition { Position = 333, Length = 6, Name = "nigata_heichi_chaku_kaisu_1", IsPrimaryKey = false, Comment = "新潟平地着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 339, Length = 6, Name = "nigata_heichi_chaku_kaisu_2", IsPrimaryKey = false, Comment = "新潟平地着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 345, Length = 6, Name = "nigata_heichi_chaku_kaisu_3", IsPrimaryKey = false, Comment = "新潟平地着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 351, Length = 6, Name = "nigata_heichi_chaku_kaisu_4", IsPrimaryKey = false, Comment = "新潟平地着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 357, Length = 6, Name = "nigata_heichi_chaku_kaisu_5", IsPrimaryKey = false, Comment = "新潟平地着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 363, Length = 6, Name = "nigata_heichi_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "新潟平地着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 369, Length = 6, Name = "nigata_shogai_chaku_kaisu_1", IsPrimaryKey = false, Comment = "新潟障害着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 375, Length = 6, Name = "nigata_shogai_chaku_kaisu_2", IsPrimaryKey = false, Comment = "新潟障害着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 381, Length = 6, Name = "nigata_shogai_chaku_kaisu_3", IsPrimaryKey = false, Comment = "新潟障害着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 387, Length = 6, Name = "nigata_shogai_chaku_kaisu_4", IsPrimaryKey = false, Comment = "新潟障害着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 393, Length = 6, Name = "nigata_shogai_chaku_kaisu_5", IsPrimaryKey = false, Comment = "新潟障害着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 399, Length = 6, Name = "nigata_shogai_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "新潟障害着回数着外", DataType = "CHAR" },
                                    
                                    // 競馬場別着回数 (東京)
                                    new NormalFieldDefinition { Position = 405, Length = 6, Name = "tokyo_heichi_chaku_kaisu_1", IsPrimaryKey = false, Comment = "東京平地着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 411, Length = 6, Name = "tokyo_heichi_chaku_kaisu_2", IsPrimaryKey = false, Comment = "東京平地着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 417, Length = 6, Name = "tokyo_heichi_chaku_kaisu_3", IsPrimaryKey = false, Comment = "東京平地着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 423, Length = 6, Name = "tokyo_heichi_chaku_kaisu_4", IsPrimaryKey = false, Comment = "東京平地着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 429, Length = 6, Name = "tokyo_heichi_chaku_kaisu_5", IsPrimaryKey = false, Comment = "東京平地着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 435, Length = 6, Name = "tokyo_heichi_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "東京平地着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 441, Length = 6, Name = "tokyo_shogai_chaku_kaisu_1", IsPrimaryKey = false, Comment = "東京障害着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 447, Length = 6, Name = "tokyo_shogai_chaku_kaisu_2", IsPrimaryKey = false, Comment = "東京障害着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 453, Length = 6, Name = "tokyo_shogai_chaku_kaisu_3", IsPrimaryKey = false, Comment = "東京障害着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 459, Length = 6, Name = "tokyo_shogai_chaku_kaisu_4", IsPrimaryKey = false, Comment = "東京障害着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 465, Length = 6, Name = "tokyo_shogai_chaku_kaisu_5", IsPrimaryKey = false, Comment = "東京障害着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 471, Length = 6, Name = "tokyo_shogai_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "東京障害着回数着外", DataType = "CHAR" },
                                    
                                    // 競馬場別着回数 (中山)
                                    new NormalFieldDefinition { Position = 477, Length = 6, Name = "nakayama_heichi_chaku_kaisu_1", IsPrimaryKey = false, Comment = "中山平地着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 483, Length = 6, Name = "nakayama_heichi_chaku_kaisu_2", IsPrimaryKey = false, Comment = "中山平地着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 489, Length = 6, Name = "nakayama_heichi_chaku_kaisu_3", IsPrimaryKey = false, Comment = "中山平地着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 495, Length = 6, Name = "nakayama_heichi_chaku_kaisu_4", IsPrimaryKey = false, Comment = "中山平地着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 501, Length = 6, Name = "nakayama_heichi_chaku_kaisu_5", IsPrimaryKey = false, Comment = "中山平地着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 507, Length = 6, Name = "nakayama_heichi_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "中山平地着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 513, Length = 6, Name = "nakayama_shogai_chaku_kaisu_1", IsPrimaryKey = false, Comment = "中山障害着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 519, Length = 6, Name = "nakayama_shogai_chaku_kaisu_2", IsPrimaryKey = false, Comment = "中山障害着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 525, Length = 6, Name = "nakayama_shogai_chaku_kaisu_3", IsPrimaryKey = false, Comment = "中山障害着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 531, Length = 6, Name = "nakayama_shogai_chaku_kaisu_4", IsPrimaryKey = false, Comment = "中山障害着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 537, Length = 6, Name = "nakayama_shogai_chaku_kaisu_5", IsPrimaryKey = false, Comment = "中山障害着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 543, Length = 6, Name = "nakayama_shogai_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "中山障害着回数着外", DataType = "CHAR" },
                                    
                                    // 競馬場別着回数 (中京)
                                    new NormalFieldDefinition { Position = 549, Length = 6, Name = "chukyo_heichi_chaku_kaisu_1", IsPrimaryKey = false, Comment = "中京平地着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 555, Length = 6, Name = "chukyo_heichi_chaku_kaisu_2", IsPrimaryKey = false, Comment = "中京平地着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 561, Length = 6, Name = "chukyo_heichi_chaku_kaisu_3", IsPrimaryKey = false, Comment = "中京平地着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 567, Length = 6, Name = "chukyo_heichi_chaku_kaisu_4", IsPrimaryKey = false, Comment = "中京平地着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 573, Length = 6, Name = "chukyo_heichi_chaku_kaisu_5", IsPrimaryKey = false, Comment = "中京平地着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 579, Length = 6, Name = "chukyo_heichi_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "中京平地着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 585, Length = 6, Name = "chukyo_shogai_chaku_kaisu_1", IsPrimaryKey = false, Comment = "中京障害着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 591, Length = 6, Name = "chukyo_shogai_chaku_kaisu_2", IsPrimaryKey = false, Comment = "中京障害着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 597, Length = 6, Name = "chukyo_shogai_chaku_kaisu_3", IsPrimaryKey = false, Comment = "中京障害着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 603, Length = 6, Name = "chukyo_shogai_chaku_kaisu_4", IsPrimaryKey = false, Comment = "中京障害着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 609, Length = 6, Name = "chukyo_shogai_chaku_kaisu_5", IsPrimaryKey = false, Comment = "中京障害着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 615, Length = 6, Name = "chukyo_shogai_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "中京障害着回数着外", DataType = "CHAR" },
                                    
                                    // 競馬場別着回数 (京都)
                                    new NormalFieldDefinition { Position = 621, Length = 6, Name = "kyoto_heichi_chaku_kaisu_1", IsPrimaryKey = false, Comment = "京都平地着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 627, Length = 6, Name = "kyoto_heichi_chaku_kaisu_2", IsPrimaryKey = false, Comment = "京都平地着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 633, Length = 6, Name = "kyoto_heichi_chaku_kaisu_3", IsPrimaryKey = false, Comment = "京都平地着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 639, Length = 6, Name = "kyoto_heichi_chaku_kaisu_4", IsPrimaryKey = false, Comment = "京都平地着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 645, Length = 6, Name = "kyoto_heichi_chaku_kaisu_5", IsPrimaryKey = false, Comment = "京都平地着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 651, Length = 6, Name = "kyoto_heichi_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "京都平地着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 657, Length = 6, Name = "kyoto_shogai_chaku_kaisu_1", IsPrimaryKey = false, Comment = "京都障害着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 663, Length = 6, Name = "kyoto_shogai_chaku_kaisu_2", IsPrimaryKey = false, Comment = "京都障害着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 669, Length = 6, Name = "kyoto_shogai_chaku_kaisu_3", IsPrimaryKey = false, Comment = "京都障害着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 675, Length = 6, Name = "kyoto_shogai_chaku_kaisu_4", IsPrimaryKey = false, Comment = "京都障害着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 681, Length = 6, Name = "kyoto_shogai_chaku_kaisu_5", IsPrimaryKey = false, Comment = "京都障害着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 687, Length = 6, Name = "kyoto_shogai_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "京都障害着回数着外", DataType = "CHAR" },
                                    
                                    // 競馬場別着回数 (阪神)
                                    new NormalFieldDefinition { Position = 693, Length = 6, Name = "hanshin_heichi_chaku_kaisu_1", IsPrimaryKey = false, Comment = "阪神平地着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 699, Length = 6, Name = "hanshin_heichi_chaku_kaisu_2", IsPrimaryKey = false, Comment = "阪神平地着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 705, Length = 6, Name = "hanshin_heichi_chaku_kaisu_3", IsPrimaryKey = false, Comment = "阪神平地着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 711, Length = 6, Name = "hanshin_heichi_chaku_kaisu_4", IsPrimaryKey = false, Comment = "阪神平地着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 717, Length = 6, Name = "hanshin_heichi_chaku_kaisu_5", IsPrimaryKey = false, Comment = "阪神平地着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 723, Length = 6, Name = "hanshin_heichi_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "阪神平地着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 729, Length = 6, Name = "hanshin_shogai_chaku_kaisu_1", IsPrimaryKey = false, Comment = "阪神障害着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 735, Length = 6, Name = "hanshin_shogai_chaku_kaisu_2", IsPrimaryKey = false, Comment = "阪神障害着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 741, Length = 6, Name = "hanshin_shogai_chaku_kaisu_3", IsPrimaryKey = false, Comment = "阪神障害着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 747, Length = 6, Name = "hanshin_shogai_chaku_kaisu_4", IsPrimaryKey = false, Comment = "阪神障害着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 753, Length = 6, Name = "hanshin_shogai_chaku_kaisu_5", IsPrimaryKey = false, Comment = "阪神障害着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 759, Length = 6, Name = "hanshin_shogai_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "阪神障害着回数着外", DataType = "CHAR" },
                                    
                                    // 競馬場別着回数 (小倉)
                                    new NormalFieldDefinition { Position = 765, Length = 6, Name = "kokura_heichi_chaku_kaisu_1", IsPrimaryKey = false, Comment = "小倉平地着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 771, Length = 6, Name = "kokura_heichi_chaku_kaisu_2", IsPrimaryKey = false, Comment = "小倉平地着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 777, Length = 6, Name = "kokura_heichi_chaku_kaisu_3", IsPrimaryKey = false, Comment = "小倉平地着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 783, Length = 6, Name = "kokura_heichi_chaku_kaisu_4", IsPrimaryKey = false, Comment = "小倉平地着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 789, Length = 6, Name = "kokura_heichi_chaku_kaisu_5", IsPrimaryKey = false, Comment = "小倉平地着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 795, Length = 6, Name = "kokura_heichi_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "小倉平地着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 801, Length = 6, Name = "kokura_shogai_chaku_kaisu_1", IsPrimaryKey = false, Comment = "小倉障害着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 807, Length = 6, Name = "kokura_shogai_chaku_kaisu_2", IsPrimaryKey = false, Comment = "小倉障害着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 813, Length = 6, Name = "kokura_shogai_chaku_kaisu_3", IsPrimaryKey = false, Comment = "小倉障害着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 819, Length = 6, Name = "kokura_shogai_chaku_kaisu_4", IsPrimaryKey = false, Comment = "小倉障害着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 825, Length = 6, Name = "kokura_shogai_chaku_kaisu_5", IsPrimaryKey = false, Comment = "小倉障害着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 831, Length = 6, Name = "kokura_shogai_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "小倉障害着回数着外", DataType = "CHAR" },
                                    
                                    // 距離別着回数
                                    new NormalFieldDefinition { Position = 837, Length = 6, Name = "shiba_1600_ika_chaku_kaisu_1", IsPrimaryKey = false, Comment = "芝1600以下着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 843, Length = 6, Name = "shiba_1600_ika_chaku_kaisu_2", IsPrimaryKey = false, Comment = "芝1600以下着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 849, Length = 6, Name = "shiba_1600_ika_chaku_kaisu_3", IsPrimaryKey = false, Comment = "芝1600以下着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 855, Length = 6, Name = "shiba_1600_ika_chaku_kaisu_4", IsPrimaryKey = false, Comment = "芝1600以下着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 861, Length = 6, Name = "shiba_1600_ika_chaku_kaisu_5", IsPrimaryKey = false, Comment = "芝1600以下着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 867, Length = 6, Name = "shiba_1600_ika_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "芝1600以下着回数着外", DataType = "CHAR" },
                                    
                                    new NormalFieldDefinition { Position = 873, Length = 6, Name = "shiba_1601_2200_chaku_kaisu_1", IsPrimaryKey = false, Comment = "芝2200以下着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 879, Length = 6, Name = "shiba_1601_2200_chaku_kaisu_2", IsPrimaryKey = false, Comment = "芝2200以下着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 885, Length = 6, Name = "shiba_1601_2200_chaku_kaisu_3", IsPrimaryKey = false, Comment = "芝2200以下着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 891, Length = 6, Name = "shiba_1601_2200_chaku_kaisu_4", IsPrimaryKey = false, Comment = "芝2200以下着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 897, Length = 6, Name = "shiba_1601_2200_chaku_kaisu_5", IsPrimaryKey = false, Comment = "芝2200以下着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 903, Length = 6, Name = "shiba_1601_2200_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "芝2200以下着回数着外", DataType = "CHAR" },
                                    
                                    new NormalFieldDefinition { Position = 909, Length = 6, Name = "shiba_2201_ijo_chaku_kaisu_1", IsPrimaryKey = false, Comment = "芝2201以上着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 915, Length = 6, Name = "shiba_2201_ijo_chaku_kaisu_2", IsPrimaryKey = false, Comment = "芝2201以上着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 921, Length = 6, Name = "shiba_2201_ijo_chaku_kaisu_3", IsPrimaryKey = false, Comment = "芝2201以上着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 927, Length = 6, Name = "shiba_2201_ijo_chaku_kaisu_4", IsPrimaryKey = false, Comment = "芝2201以上着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 933, Length = 6, Name = "shiba_2201_ijo_chaku_kaisu_5", IsPrimaryKey = false, Comment = "芝2201以上着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 939, Length = 6, Name = "shiba_2201_ijo_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "芝2201以上着回数着外", DataType = "CHAR" },
                                    
                                    new NormalFieldDefinition { Position = 945, Length = 6, Name = "dirt_1600_ika_chaku_kaisu_1", IsPrimaryKey = false, Comment = "ダート1600以下着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 951, Length = 6, Name = "dirt_1600_ika_chaku_kaisu_2", IsPrimaryKey = false, Comment = "ダート1600以下着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 957, Length = 6, Name = "dirt_1600_ika_chaku_kaisu_3", IsPrimaryKey = false, Comment = "ダート1600以下着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 963, Length = 6, Name = "dirt_1600_ika_chaku_kaisu_4", IsPrimaryKey = false, Comment = "ダート1600以下着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 969, Length = 6, Name = "dirt_1600_ika_chaku_kaisu_5", IsPrimaryKey = false, Comment = "ダート1600以下着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 975, Length = 6, Name = "dirt_1600_ika_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "ダート1600以下着回数着外", DataType = "CHAR" },
                                    
                                    new NormalFieldDefinition { Position = 981, Length = 6, Name = "dirt_1601_2200_chaku_kaisu_1", IsPrimaryKey = false, Comment = "ダート2200以下着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 987, Length = 6, Name = "dirt_1601_2200_chaku_kaisu_2", IsPrimaryKey = false, Comment = "ダート2200以下着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 993, Length = 6, Name = "dirt_1601_2200_chaku_kaisu_3", IsPrimaryKey = false, Comment = "ダート2200以下着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 999, Length = 6, Name = "dirt_1601_2200_chaku_kaisu_4", IsPrimaryKey = false, Comment = "ダート2200以下着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1005, Length = 6, Name = "dirt_1601_2200_chaku_kaisu_5", IsPrimaryKey = false, Comment = "ダート2200以下着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1011, Length = 6, Name = "dirt_1601_2200_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "ダート2200以下着回数着外", DataType = "CHAR" },
                                    
                                    new NormalFieldDefinition { Position = 1017, Length = 6, Name = "dirt_2201_ijo_chaku_kaisu_1", IsPrimaryKey = false, Comment = "ダート2201以上着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1023, Length = 6, Name = "dirt_2201_ijo_chaku_kaisu_2", IsPrimaryKey = false, Comment = "ダート2201以上着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1029, Length = 6, Name = "dirt_2201_ijo_chaku_kaisu_3", IsPrimaryKey = false, Comment = "ダート2201以上着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1035, Length = 6, Name = "dirt_2201_ijo_chaku_kaisu_4", IsPrimaryKey = false, Comment = "ダート2201以上着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1041, Length = 6, Name = "dirt_2201_ijo_chaku_kaisu_5", IsPrimaryKey = false, Comment = "ダート2201以上着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1047, Length = 6, Name = "dirt_2201_ijo_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "ダート2201以上着回数着外", DataType = "CHAR" }
                                }
                            }
                        }
                    }
                }
            };
            recordDefinitions.Add(ksRecord);

            // CH - 調教師マスタ
            var chRecord = new RecordDefinition
            {
                RecordTypeId = "CH",
                CreationDateField = new FieldDefinition { Position = 4, Length = 8 },
                Table = new TableDefinition
                {
                    Name = "chokyoshi_master",
                    Comment = "調教師マスタ",
                    Fields = new List<FieldDefinition>
                    {
                        new NormalFieldDefinition { Position = 1, Length = 2, Name = "record_type_id", IsPrimaryKey = false, Comment = "レコード種別ID", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 3, Length = 1, Name = "data_kubun", IsPrimaryKey = false, Comment = "データ区分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 12, Length = 5, Name = "chokyoshi_code", IsPrimaryKey = true, Comment = "調教師コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 17, Length = 1, Name = "chokyoshi_massho_kubun", IsPrimaryKey = false, Comment = "調教師抹消区分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 18, Length = 8, Name = "chokyoshi_menkyo_kofu_date", IsPrimaryKey = false, Comment = "調教師免許交付年月日", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 26, Length = 8, Name = "chokyoshi_menkyo_massho_date", IsPrimaryKey = false, Comment = "調教師免許抹消年月日", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 34, Length = 8, Name = "birth_date", IsPrimaryKey = false, Comment = "生年月日", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 42, Length = 34, Name = "chokyoshi_name", IsPrimaryKey = false, Comment = "調教師名", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 76, Length = 30, Name = "chokyoshi_name_hankaku_kana", IsPrimaryKey = false, Comment = "調教師名半角カナ", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 106, Length = 8, Name = "chokyoshi_name_short", IsPrimaryKey = false, Comment = "調教師名略称", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 114, Length = 80, Name = "chokyoshi_name_eng", IsPrimaryKey = false, Comment = "調教師名欧字", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 194, Length = 1, Name = "seibetsu_kubun", IsPrimaryKey = false, Comment = "性別区分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 195, Length = 1, Name = "chokyoshi_tozai_shozoku_code", IsPrimaryKey = false, Comment = "調教師東西所属コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 196, Length = 20, Name = "shotai_chiiki_name", IsPrimaryKey = false, Comment = "招待地域名", DataType = "CHAR" },
                        
                        // 最近重賞勝利情報 (3回繰り返し)
                        new RepeatFieldDefinition
                        {
                            Position = 216,
                            Length = 163,
                            RepeatCount = 3,
                            Table = new TableDefinition
                            {
                                Name = "saikin_jusho_shori_joho",
                                Comment = "最近重賞勝利情報",
                                Fields = new List<FieldDefinition>
                                {
                                    new NormalFieldDefinition { Position = 1, Length = 16, Name = "race_id", IsPrimaryKey = false, Comment = "年月日場回日R", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 17, Length = 60, Name = "kyoso_name_hondai", IsPrimaryKey = false, Comment = "競走名本題", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 77, Length = 20, Name = "kyoso_name_short_10", IsPrimaryKey = false, Comment = "競走名略称10文字", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 97, Length = 12, Name = "kyoso_name_short_6", IsPrimaryKey = false, Comment = "競走名略称6文字", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 109, Length = 6, Name = "kyoso_name_short_3", IsPrimaryKey = false, Comment = "競走名略称3文字", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 115, Length = 1, Name = "grade_code", IsPrimaryKey = false, Comment = "グレードコード", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 116, Length = 2, Name = "shusso_tosu", IsPrimaryKey = false, Comment = "出走頭数", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 118, Length = 10, Name = "ketto_toroku_number", IsPrimaryKey = false, Comment = "血統登録番号", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 128, Length = 36, Name = "uma_name", IsPrimaryKey = false, Comment = "馬名", DataType = "CHAR" }
                                }
                            }
                        },
                        
                        // 本年・前年・累計成績情報 (3回繰り返し)
                        new RepeatFieldDefinition
                        {
                            Position = 705,
                            Length = 1052,
                            RepeatCount = 3,
                            Table = new TableDefinition
                            {
                                Name = "seiseki_joho",
                                Comment = "本年・前年・累計成績情報",
                                Fields = new List<FieldDefinition>
                                {
                                    new NormalFieldDefinition { Position = 1, Length = 4, Name = "settei_year", IsPrimaryKey = false, Comment = "設定年", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 5, Length = 10, Name = "heichi_hon_shokin_goukei", IsPrimaryKey = false, Comment = "平地本賞金合計", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 15, Length = 10, Name = "shogai_hon_shokin_goukei", IsPrimaryKey = false, Comment = "障害本賞金合計", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 25, Length = 10, Name = "heichi_fuka_shokin_goukei", IsPrimaryKey = false, Comment = "平地付加賞金合計", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 35, Length = 10, Name = "shogai_fuka_shokin_goukei", IsPrimaryKey = false, Comment = "障害付加賞金合計", DataType = "CHAR" },
                                    
                                    // 平地着回数 (1着～5着及び着外の6回)
                                    new NormalFieldDefinition { Position = 45, Length = 6, Name = "heichi_chaku_kaisu_1", IsPrimaryKey = false, Comment = "平地着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 51, Length = 6, Name = "heichi_chaku_kaisu_2", IsPrimaryKey = false, Comment = "平地着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 57, Length = 6, Name = "heichi_chaku_kaisu_3", IsPrimaryKey = false, Comment = "平地着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 63, Length = 6, Name = "heichi_chaku_kaisu_4", IsPrimaryKey = false, Comment = "平地着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 69, Length = 6, Name = "heichi_chaku_kaisu_5", IsPrimaryKey = false, Comment = "平地着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 75, Length = 6, Name = "heichi_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "平地着回数着外", DataType = "CHAR" },
                                    
                                    // 障害着回数 (1着～5着及び着外の6回)
                                    new NormalFieldDefinition { Position = 81, Length = 6, Name = "shogai_chaku_kaisu_1", IsPrimaryKey = false, Comment = "障害着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 87, Length = 6, Name = "shogai_chaku_kaisu_2", IsPrimaryKey = false, Comment = "障害着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 93, Length = 6, Name = "shogai_chaku_kaisu_3", IsPrimaryKey = false, Comment = "障害着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 99, Length = 6, Name = "shogai_chaku_kaisu_4", IsPrimaryKey = false, Comment = "障害着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 105, Length = 6, Name = "shogai_chaku_kaisu_5", IsPrimaryKey = false, Comment = "障害着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 111, Length = 6, Name = "shogai_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "障害着回数着外", DataType = "CHAR" },
                                    
                                    // 競馬場別着回数 - 札幌平地
                                    new NormalFieldDefinition { Position = 117, Length = 6, Name = "sapporo_heichi_chaku_kaisu_1", IsPrimaryKey = false, Comment = "札幌平地着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 123, Length = 6, Name = "sapporo_heichi_chaku_kaisu_2", IsPrimaryKey = false, Comment = "札幌平地着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 129, Length = 6, Name = "sapporo_heichi_chaku_kaisu_3", IsPrimaryKey = false, Comment = "札幌平地着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 135, Length = 6, Name = "sapporo_heichi_chaku_kaisu_4", IsPrimaryKey = false, Comment = "札幌平地着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 141, Length = 6, Name = "sapporo_heichi_chaku_kaisu_5", IsPrimaryKey = false, Comment = "札幌平地着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 147, Length = 6, Name = "sapporo_heichi_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "札幌平地着回数着外", DataType = "CHAR" },
                                    
                                    // 競馬場別着回数 - 札幌障害
                                    new NormalFieldDefinition { Position = 153, Length = 6, Name = "sapporo_shogai_chaku_kaisu_1", IsPrimaryKey = false, Comment = "札幌障害着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 159, Length = 6, Name = "sapporo_shogai_chaku_kaisu_2", IsPrimaryKey = false, Comment = "札幌障害着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 165, Length = 6, Name = "sapporo_shogai_chaku_kaisu_3", IsPrimaryKey = false, Comment = "札幌障害着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 171, Length = 6, Name = "sapporo_shogai_chaku_kaisu_4", IsPrimaryKey = false, Comment = "札幌障害着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 177, Length = 6, Name = "sapporo_shogai_chaku_kaisu_5", IsPrimaryKey = false, Comment = "札幌障害着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 183, Length = 6, Name = "sapporo_shogai_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "札幌障害着回数着外", DataType = "CHAR" },
                                    
                                    // 競馬場別着回数 - 函館平地
                                    new NormalFieldDefinition { Position = 189, Length = 6, Name = "hakodate_heichi_chaku_kaisu_1", IsPrimaryKey = false, Comment = "函館平地着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 195, Length = 6, Name = "hakodate_heichi_chaku_kaisu_2", IsPrimaryKey = false, Comment = "函館平地着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 201, Length = 6, Name = "hakodate_heichi_chaku_kaisu_3", IsPrimaryKey = false, Comment = "函館平地着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 207, Length = 6, Name = "hakodate_heichi_chaku_kaisu_4", IsPrimaryKey = false, Comment = "函館平地着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 213, Length = 6, Name = "hakodate_heichi_chaku_kaisu_5", IsPrimaryKey = false, Comment = "函館平地着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 219, Length = 6, Name = "hakodate_heichi_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "函館平地着回数着外", DataType = "CHAR" },
                                    
                                    // 競馬場別着回数 - 函館障害
                                    new NormalFieldDefinition { Position = 225, Length = 6, Name = "hakodate_shogai_chaku_kaisu_1", IsPrimaryKey = false, Comment = "函館障害着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 231, Length = 6, Name = "hakodate_shogai_chaku_kaisu_2", IsPrimaryKey = false, Comment = "函館障害着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 237, Length = 6, Name = "hakodate_shogai_chaku_kaisu_3", IsPrimaryKey = false, Comment = "函館障害着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 243, Length = 6, Name = "hakodate_shogai_chaku_kaisu_4", IsPrimaryKey = false, Comment = "函館障害着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 249, Length = 6, Name = "hakodate_shogai_chaku_kaisu_5", IsPrimaryKey = false, Comment = "函館障害着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 255, Length = 6, Name = "hakodate_shogai_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "函館障害着回数着外", DataType = "CHAR" },
                                    
                                    // 競馬場別着回数 - 福島平地
                                    new NormalFieldDefinition { Position = 261, Length = 6, Name = "fukushima_heichi_chaku_kaisu_1", IsPrimaryKey = false, Comment = "福島平地着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 267, Length = 6, Name = "fukushima_heichi_chaku_kaisu_2", IsPrimaryKey = false, Comment = "福島平地着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 273, Length = 6, Name = "fukushima_heichi_chaku_kaisu_3", IsPrimaryKey = false, Comment = "福島平地着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 279, Length = 6, Name = "fukushima_heichi_chaku_kaisu_4", IsPrimaryKey = false, Comment = "福島平地着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 285, Length = 6, Name = "fukushima_heichi_chaku_kaisu_5", IsPrimaryKey = false, Comment = "福島平地着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 291, Length = 6, Name = "fukushima_heichi_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "福島平地着回数着外", DataType = "CHAR" },
                                    
                                    // 競馬場別着回数 - 福島障害
                                    new NormalFieldDefinition { Position = 297, Length = 6, Name = "fukushima_shogai_chaku_kaisu_1", IsPrimaryKey = false, Comment = "福島障害着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 303, Length = 6, Name = "fukushima_shogai_chaku_kaisu_2", IsPrimaryKey = false, Comment = "福島障害着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 309, Length = 6, Name = "fukushima_shogai_chaku_kaisu_3", IsPrimaryKey = false, Comment = "福島障害着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 315, Length = 6, Name = "fukushima_shogai_chaku_kaisu_4", IsPrimaryKey = false, Comment = "福島障害着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 321, Length = 6, Name = "fukushima_shogai_chaku_kaisu_5", IsPrimaryKey = false, Comment = "福島障害着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 327, Length = 6, Name = "fukushima_shogai_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "福島障害着回数着外", DataType = "CHAR" },
                                    
                                    // 競馬場別着回数 - 新潟平地
                                    new NormalFieldDefinition { Position = 333, Length = 6, Name = "nigata_heichi_chaku_kaisu_1", IsPrimaryKey = false, Comment = "新潟平地着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 339, Length = 6, Name = "nigata_heichi_chaku_kaisu_2", IsPrimaryKey = false, Comment = "新潟平地着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 345, Length = 6, Name = "nigata_heichi_chaku_kaisu_3", IsPrimaryKey = false, Comment = "新潟平地着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 351, Length = 6, Name = "nigata_heichi_chaku_kaisu_4", IsPrimaryKey = false, Comment = "新潟平地着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 357, Length = 6, Name = "nigata_heichi_chaku_kaisu_5", IsPrimaryKey = false, Comment = "新潟平地着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 363, Length = 6, Name = "nigata_heichi_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "新潟平地着回数着外", DataType = "CHAR" },
                                    
                                    // 競馬場別着回数 - 新潟障害
                                    new NormalFieldDefinition { Position = 369, Length = 6, Name = "nigata_shogai_chaku_kaisu_1", IsPrimaryKey = false, Comment = "新潟障害着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 375, Length = 6, Name = "nigata_shogai_chaku_kaisu_2", IsPrimaryKey = false, Comment = "新潟障害着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 381, Length = 6, Name = "nigata_shogai_chaku_kaisu_3", IsPrimaryKey = false, Comment = "新潟障害着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 387, Length = 6, Name = "nigata_shogai_chaku_kaisu_4", IsPrimaryKey = false, Comment = "新潟障害着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 393, Length = 6, Name = "nigata_shogai_chaku_kaisu_5", IsPrimaryKey = false, Comment = "新潟障害着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 399, Length = 6, Name = "nigata_shogai_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "新潟障害着回数着外", DataType = "CHAR" },
                                    
                                    // 競馬場別着回数 - 東京平地
                                    new NormalFieldDefinition { Position = 405, Length = 6, Name = "tokyo_heichi_chaku_kaisu_1", IsPrimaryKey = false, Comment = "東京平地着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 411, Length = 6, Name = "tokyo_heichi_chaku_kaisu_2", IsPrimaryKey = false, Comment = "東京平地着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 417, Length = 6, Name = "tokyo_heichi_chaku_kaisu_3", IsPrimaryKey = false, Comment = "東京平地着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 423, Length = 6, Name = "tokyo_heichi_chaku_kaisu_4", IsPrimaryKey = false, Comment = "東京平地着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 429, Length = 6, Name = "tokyo_heichi_chaku_kaisu_5", IsPrimaryKey = false, Comment = "東京平地着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 435, Length = 6, Name = "tokyo_heichi_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "東京平地着回数着外", DataType = "CHAR" },
                                    
                                    // 競馬場別着回数 - 東京障害
                                    new NormalFieldDefinition { Position = 441, Length = 6, Name = "tokyo_shogai_chaku_kaisu_1", IsPrimaryKey = false, Comment = "東京障害着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 447, Length = 6, Name = "tokyo_shogai_chaku_kaisu_2", IsPrimaryKey = false, Comment = "東京障害着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 453, Length = 6, Name = "tokyo_shogai_chaku_kaisu_3", IsPrimaryKey = false, Comment = "東京障害着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 459, Length = 6, Name = "tokyo_shogai_chaku_kaisu_4", IsPrimaryKey = false, Comment = "東京障害着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 465, Length = 6, Name = "tokyo_shogai_chaku_kaisu_5", IsPrimaryKey = false, Comment = "東京障害着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 471, Length = 6, Name = "tokyo_shogai_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "東京障害着回数着外", DataType = "CHAR" },
                                    
                                    // 競馬場別着回数 - 中山平地
                                    new NormalFieldDefinition { Position = 477, Length = 6, Name = "nakayama_heichi_chaku_kaisu_1", IsPrimaryKey = false, Comment = "中山平地着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 483, Length = 6, Name = "nakayama_heichi_chaku_kaisu_2", IsPrimaryKey = false, Comment = "中山平地着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 489, Length = 6, Name = "nakayama_heichi_chaku_kaisu_3", IsPrimaryKey = false, Comment = "中山平地着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 495, Length = 6, Name = "nakayama_heichi_chaku_kaisu_4", IsPrimaryKey = false, Comment = "中山平地着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 501, Length = 6, Name = "nakayama_heichi_chaku_kaisu_5", IsPrimaryKey = false, Comment = "中山平地着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 507, Length = 6, Name = "nakayama_heichi_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "中山平地着回数着外", DataType = "CHAR" },
                                    
                                    // 競馬場別着回数 - 中山障害
                                    new NormalFieldDefinition { Position = 513, Length = 6, Name = "nakayama_shogai_chaku_kaisu_1", IsPrimaryKey = false, Comment = "中山障害着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 519, Length = 6, Name = "nakayama_shogai_chaku_kaisu_2", IsPrimaryKey = false, Comment = "中山障害着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 525, Length = 6, Name = "nakayama_shogai_chaku_kaisu_3", IsPrimaryKey = false, Comment = "中山障害着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 531, Length = 6, Name = "nakayama_shogai_chaku_kaisu_4", IsPrimaryKey = false, Comment = "中山障害着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 537, Length = 6, Name = "nakayama_shogai_chaku_kaisu_5", IsPrimaryKey = false, Comment = "中山障害着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 543, Length = 6, Name = "nakayama_shogai_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "中山障害着回数着外", DataType = "CHAR" },
                                    
                                    // 競馬場別着回数 - 中京平地
                                    new NormalFieldDefinition { Position = 549, Length = 6, Name = "chukyo_heichi_chaku_kaisu_1", IsPrimaryKey = false, Comment = "中京平地着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 555, Length = 6, Name = "chukyo_heichi_chaku_kaisu_2", IsPrimaryKey = false, Comment = "中京平地着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 561, Length = 6, Name = "chukyo_heichi_chaku_kaisu_3", IsPrimaryKey = false, Comment = "中京平地着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 567, Length = 6, Name = "chukyo_heichi_chaku_kaisu_4", IsPrimaryKey = false, Comment = "中京平地着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 573, Length = 6, Name = "chukyo_heichi_chaku_kaisu_5", IsPrimaryKey = false, Comment = "中京平地着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 579, Length = 6, Name = "chukyo_heichi_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "中京平地着回数着外", DataType = "CHAR" },
                                    
                                    // 競馬場別着回数 - 中京障害
                                    new NormalFieldDefinition { Position = 585, Length = 6, Name = "chukyo_shogai_chaku_kaisu_1", IsPrimaryKey = false, Comment = "中京障害着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 591, Length = 6, Name = "chukyo_shogai_chaku_kaisu_2", IsPrimaryKey = false, Comment = "中京障害着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 597, Length = 6, Name = "chukyo_shogai_chaku_kaisu_3", IsPrimaryKey = false, Comment = "中京障害着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 603, Length = 6, Name = "chukyo_shogai_chaku_kaisu_4", IsPrimaryKey = false, Comment = "中京障害着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 609, Length = 6, Name = "chukyo_shogai_chaku_kaisu_5", IsPrimaryKey = false, Comment = "中京障害着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 615, Length = 6, Name = "chukyo_shogai_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "中京障害着回数着外", DataType = "CHAR" },
                                    
                                    // 競馬場別着回数 - 京都平地
                                    new NormalFieldDefinition { Position = 621, Length = 6, Name = "kyoto_heichi_chaku_kaisu_1", IsPrimaryKey = false, Comment = "京都平地着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 627, Length = 6, Name = "kyoto_heichi_chaku_kaisu_2", IsPrimaryKey = false, Comment = "京都平地着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 633, Length = 6, Name = "kyoto_heichi_chaku_kaisu_3", IsPrimaryKey = false, Comment = "京都平地着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 639, Length = 6, Name = "kyoto_heichi_chaku_kaisu_4", IsPrimaryKey = false, Comment = "京都平地着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 645, Length = 6, Name = "kyoto_heichi_chaku_kaisu_5", IsPrimaryKey = false, Comment = "京都平地着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 651, Length = 6, Name = "kyoto_heichi_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "京都平地着回数着外", DataType = "CHAR" },
                                    
                                    // 競馬場別着回数 - 京都障害
                                    new NormalFieldDefinition { Position = 657, Length = 6, Name = "kyoto_shogai_chaku_kaisu_1", IsPrimaryKey = false, Comment = "京都障害着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 663, Length = 6, Name = "kyoto_shogai_chaku_kaisu_2", IsPrimaryKey = false, Comment = "京都障害着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 669, Length = 6, Name = "kyoto_shogai_chaku_kaisu_3", IsPrimaryKey = false, Comment = "京都障害着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 675, Length = 6, Name = "kyoto_shogai_chaku_kaisu_4", IsPrimaryKey = false, Comment = "京都障害着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 681, Length = 6, Name = "kyoto_shogai_chaku_kaisu_5", IsPrimaryKey = false, Comment = "京都障害着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 687, Length = 6, Name = "kyoto_shogai_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "京都障害着回数着外", DataType = "CHAR" },
                                    
                                    // 競馬場別着回数 - 阪神平地
                                    new NormalFieldDefinition { Position = 693, Length = 6, Name = "hanshin_heichi_chaku_kaisu_1", IsPrimaryKey = false, Comment = "阪神平地着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 699, Length = 6, Name = "hanshin_heichi_chaku_kaisu_2", IsPrimaryKey = false, Comment = "阪神平地着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 705, Length = 6, Name = "hanshin_heichi_chaku_kaisu_3", IsPrimaryKey = false, Comment = "阪神平地着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 711, Length = 6, Name = "hanshin_heichi_chaku_kaisu_4", IsPrimaryKey = false, Comment = "阪神平地着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 717, Length = 6, Name = "hanshin_heichi_chaku_kaisu_5", IsPrimaryKey = false, Comment = "阪神平地着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 723, Length = 6, Name = "hanshin_heichi_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "阪神平地着回数着外", DataType = "CHAR" },
                                    
                                    // 競馬場別着回数 - 阪神障害
                                    new NormalFieldDefinition { Position = 729, Length = 6, Name = "hanshin_shogai_chaku_kaisu_1", IsPrimaryKey = false, Comment = "阪神障害着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 735, Length = 6, Name = "hanshin_shogai_chaku_kaisu_2", IsPrimaryKey = false, Comment = "阪神障害着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 741, Length = 6, Name = "hanshin_shogai_chaku_kaisu_3", IsPrimaryKey = false, Comment = "阪神障害着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 747, Length = 6, Name = "hanshin_shogai_chaku_kaisu_4", IsPrimaryKey = false, Comment = "阪神障害着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 753, Length = 6, Name = "hanshin_shogai_chaku_kaisu_5", IsPrimaryKey = false, Comment = "阪神障害着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 759, Length = 6, Name = "hanshin_shogai_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "阪神障害着回数着外", DataType = "CHAR" },
                                    
                                    // 競馬場別着回数 - 小倉平地
                                    new NormalFieldDefinition { Position = 765, Length = 6, Name = "kokura_heichi_chaku_kaisu_1", IsPrimaryKey = false, Comment = "小倉平地着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 771, Length = 6, Name = "kokura_heichi_chaku_kaisu_2", IsPrimaryKey = false, Comment = "小倉平地着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 777, Length = 6, Name = "kokura_heichi_chaku_kaisu_3", IsPrimaryKey = false, Comment = "小倉平地着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 783, Length = 6, Name = "kokura_heichi_chaku_kaisu_4", IsPrimaryKey = false, Comment = "小倉平地着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 789, Length = 6, Name = "kokura_heichi_chaku_kaisu_5", IsPrimaryKey = false, Comment = "小倉平地着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 795, Length = 6, Name = "kokura_heichi_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "小倉平地着回数着外", DataType = "CHAR" },
                                    
                                    // 競馬場別着回数 - 小倉障害
                                    new NormalFieldDefinition { Position = 801, Length = 6, Name = "kokura_shogai_chaku_kaisu_1", IsPrimaryKey = false, Comment = "小倉障害着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 807, Length = 6, Name = "kokura_shogai_chaku_kaisu_2", IsPrimaryKey = false, Comment = "小倉障害着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 813, Length = 6, Name = "kokura_shogai_chaku_kaisu_3", IsPrimaryKey = false, Comment = "小倉障害着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 819, Length = 6, Name = "kokura_shogai_chaku_kaisu_4", IsPrimaryKey = false, Comment = "小倉障害着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 825, Length = 6, Name = "kokura_shogai_chaku_kaisu_5", IsPrimaryKey = false, Comment = "小倉障害着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 831, Length = 6, Name = "kokura_shogai_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "小倉障害着回数着外", DataType = "CHAR" },
                                    
                                    // 距離別着回数 - 芝1600m以下
                                    new NormalFieldDefinition { Position = 837, Length = 6, Name = "shiba_1600_ika_chaku_kaisu_1", IsPrimaryKey = false, Comment = "芝16下・着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 843, Length = 6, Name = "shiba_1600_ika_chaku_kaisu_2", IsPrimaryKey = false, Comment = "芝16下・着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 849, Length = 6, Name = "shiba_1600_ika_chaku_kaisu_3", IsPrimaryKey = false, Comment = "芝16下・着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 855, Length = 6, Name = "shiba_1600_ika_chaku_kaisu_4", IsPrimaryKey = false, Comment = "芝16下・着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 861, Length = 6, Name = "shiba_1600_ika_chaku_kaisu_5", IsPrimaryKey = false, Comment = "芝16下・着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 867, Length = 6, Name = "shiba_1600_ika_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "芝16下・着回数着外", DataType = "CHAR" },
                                    
                                    // 距離別着回数 - 芝1601m-2200m
                                    new NormalFieldDefinition { Position = 873, Length = 6, Name = "shiba_1601_2200_chaku_kaisu_1", IsPrimaryKey = false, Comment = "芝22下・着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 879, Length = 6, Name = "shiba_1601_2200_chaku_kaisu_2", IsPrimaryKey = false, Comment = "芝22下・着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 885, Length = 6, Name = "shiba_1601_2200_chaku_kaisu_3", IsPrimaryKey = false, Comment = "芝22下・着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 891, Length = 6, Name = "shiba_1601_2200_chaku_kaisu_4", IsPrimaryKey = false, Comment = "芝22下・着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 897, Length = 6, Name = "shiba_1601_2200_chaku_kaisu_5", IsPrimaryKey = false, Comment = "芝22下・着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 903, Length = 6, Name = "shiba_1601_2200_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "芝22下・着回数着外", DataType = "CHAR" },
                                    
                                    // 距離別着回数 - 芝2201m以上
                                    new NormalFieldDefinition { Position = 909, Length = 6, Name = "shiba_2201_ijo_chaku_kaisu_1", IsPrimaryKey = false, Comment = "芝22超・着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 915, Length = 6, Name = "shiba_2201_ijo_chaku_kaisu_2", IsPrimaryKey = false, Comment = "芝22超・着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 921, Length = 6, Name = "shiba_2201_ijo_chaku_kaisu_3", IsPrimaryKey = false, Comment = "芝22超・着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 927, Length = 6, Name = "shiba_2201_ijo_chaku_kaisu_4", IsPrimaryKey = false, Comment = "芝22超・着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 933, Length = 6, Name = "shiba_2201_ijo_chaku_kaisu_5", IsPrimaryKey = false, Comment = "芝22超・着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 939, Length = 6, Name = "shiba_2201_ijo_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "芝22超・着回数着外", DataType = "CHAR" },
                                    
                                    // 距離別着回数 - ダート1600m以下
                                    new NormalFieldDefinition { Position = 945, Length = 6, Name = "dirt_1600_ika_chaku_kaisu_1", IsPrimaryKey = false, Comment = "ダ16下・着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 951, Length = 6, Name = "dirt_1600_ika_chaku_kaisu_2", IsPrimaryKey = false, Comment = "ダ16下・着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 957, Length = 6, Name = "dirt_1600_ika_chaku_kaisu_3", IsPrimaryKey = false, Comment = "ダ16下・着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 963, Length = 6, Name = "dirt_1600_ika_chaku_kaisu_4", IsPrimaryKey = false, Comment = "ダ16下・着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 969, Length = 6, Name = "dirt_1600_ika_chaku_kaisu_5", IsPrimaryKey = false, Comment = "ダ16下・着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 975, Length = 6, Name = "dirt_1600_ika_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "ダ16下・着回数着外", DataType = "CHAR" },
                                    
                                    // 距離別着回数 - ダート1601m-2200m
                                    new NormalFieldDefinition { Position = 981, Length = 6, Name = "dirt_1601_2200_chaku_kaisu_1", IsPrimaryKey = false, Comment = "ダ22下・着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 987, Length = 6, Name = "dirt_1601_2200_chaku_kaisu_2", IsPrimaryKey = false, Comment = "ダ22下・着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 993, Length = 6, Name = "dirt_1601_2200_chaku_kaisu_3", IsPrimaryKey = false, Comment = "ダ22下・着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 999, Length = 6, Name = "dirt_1601_2200_chaku_kaisu_4", IsPrimaryKey = false, Comment = "ダ22下・着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1005, Length = 6, Name = "dirt_1601_2200_chaku_kaisu_5", IsPrimaryKey = false, Comment = "ダ22下・着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1011, Length = 6, Name = "dirt_1601_2200_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "ダ22下・着回数着外", DataType = "CHAR" },
                                    
                                    // 距離別着回数 - ダート2201m以上
                                    new NormalFieldDefinition { Position = 1017, Length = 6, Name = "dirt_2201_ijo_chaku_kaisu_1", IsPrimaryKey = false, Comment = "ダ22超・着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1023, Length = 6, Name = "dirt_2201_ijo_chaku_kaisu_2", IsPrimaryKey = false, Comment = "ダ22超・着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1029, Length = 6, Name = "dirt_2201_ijo_chaku_kaisu_3", IsPrimaryKey = false, Comment = "ダ22超・着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1035, Length = 6, Name = "dirt_2201_ijo_chaku_kaisu_4", IsPrimaryKey = false, Comment = "ダ22超・着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1041, Length = 6, Name = "dirt_2201_ijo_chaku_kaisu_5", IsPrimaryKey = false, Comment = "ダ22超・着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1047, Length = 6, Name = "dirt_2201_ijo_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "ダ22超・着回数着外", DataType = "CHAR" }
                                }
                            }
                        }
                    }
                }
            };
            recordDefinitions.Add(chRecord);

            // BR - 生産者マスタ
            var brRecord = new RecordDefinition
            {
                RecordTypeId = "BR",
                CreationDateField = new FieldDefinition { Position = 4, Length = 8 },
                Table = new TableDefinition
                {
                    Name = "seisansha_master",
                    Comment = "生産者マスタ",
                    Fields = new List<FieldDefinition>
                    {
                        new NormalFieldDefinition { Position = 1, Length = 2, Name = "record_type_id", IsPrimaryKey = false, Comment = "レコード種別ID", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 3, Length = 1, Name = "data_kubun", IsPrimaryKey = false, Comment = "データ区分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 12, Length = 8, Name = "seisansha_code", IsPrimaryKey = true, Comment = "生産者コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 20, Length = 72, Name = "seisansha_name_hojinkaku_ari", IsPrimaryKey = false, Comment = "生産者名(法人格有)", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 92, Length = 72, Name = "seisansha_name_hojinkaku_nashi", IsPrimaryKey = false, Comment = "生産者名(法人格無)", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 164, Length = 72, Name = "seisansha_name_hankaku_kana", IsPrimaryKey = false, Comment = "生産者名半角カナ", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 236, Length = 168, Name = "seisansha_name_eng", IsPrimaryKey = false, Comment = "生産者名欧字", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 404, Length = 20, Name = "seisansha_jusho_jichisho_name", IsPrimaryKey = false, Comment = "生産者住所自治省名", DataType = "CHAR" },
                        new RepeatFieldDefinition
                        {
                            Position = 424,
                            Length = 60,
                            RepeatCount = 2,
                            Table = new TableDefinition
                            {
                                Name = "seiseki_joho",
                                Comment = "本年・累計成績情報",
                                Fields = new List<FieldDefinition>
                                {
                                    new NormalFieldDefinition { Position = 1, Length = 4, Name = "settei_year", IsPrimaryKey = false, Comment = "設定年", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 5, Length = 10, Name = "hon_shokin_gokei", IsPrimaryKey = false, Comment = "本賞金合計", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 15, Length = 10, Name = "fuka_shokin_gokei", IsPrimaryKey = false, Comment = "付加賞金合計", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 25, Length = 6, Name = "chaku_kaisu_1", IsPrimaryKey = false, Comment = "着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 31, Length = 6, Name = "chaku_kaisu_2", IsPrimaryKey = false, Comment = "着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 37, Length = 6, Name = "chaku_kaisu_3", IsPrimaryKey = false, Comment = "着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 43, Length = 6, Name = "chaku_kaisu_4", IsPrimaryKey = false, Comment = "着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 49, Length = 6, Name = "chaku_kaisu_5", IsPrimaryKey = false, Comment = "着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 55, Length = 6, Name = "chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "着回数着外", DataType = "CHAR" }
                                }
                            }
                        }
                    }
                }
            };
            recordDefinitions.Add(brRecord);

            // BN - 馬主マスタ
            var bnRecord = new RecordDefinition
            {
                RecordTypeId = "BN",
                CreationDateField = new FieldDefinition { Position = 4, Length = 8 },
                Table = new TableDefinition
                {
                    Name = "banushi_master",
                    Comment = "馬主マスタ",
                    Fields = new List<FieldDefinition>
                    {
                        new NormalFieldDefinition { Position = 1, Length = 2, Name = "record_type_id", IsPrimaryKey = false, Comment = "レコード種別ID", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 3, Length = 1, Name = "data_kubun", IsPrimaryKey = false, Comment = "データ区分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 12, Length = 6, Name = "banushi_code", IsPrimaryKey = true, Comment = "馬主コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 18, Length = 64, Name = "banushi_name_hojinkaku_ari", IsPrimaryKey = false, Comment = "馬主名法人格有", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 82, Length = 64, Name = "banushi_name_hojinkaku_nashi", IsPrimaryKey = false, Comment = "馬主名法人格無", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 146, Length = 50, Name = "banushi_name_hankaku_kana", IsPrimaryKey = false, Comment = "馬主名半角カナ", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 196, Length = 100, Name = "banushi_name_eng", IsPrimaryKey = false, Comment = "馬主名欧字", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 296, Length = 60, Name = "fukushoku_hyoji", IsPrimaryKey = false, Comment = "服色標示", DataType = "CHAR" },
                        new RepeatFieldDefinition
                        {
                            Position = 356,
                            Length = 60,
                            RepeatCount = 2,
                            Table = new TableDefinition
                            {
                                Name = "seiseki_joho",
                                Comment = "本年・累計成績情報",
                                Fields = new List<FieldDefinition>
                                {
                                    new NormalFieldDefinition { Position = 1, Length = 4, Name = "settei_year", IsPrimaryKey = false, Comment = "設定年", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 5, Length = 10, Name = "hon_shokin_gokei", IsPrimaryKey = false, Comment = "本賞金合計", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 15, Length = 10, Name = "fuka_shokin_gokei", IsPrimaryKey = false, Comment = "付加賞金合計", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 25, Length = 6, Name = "chaku_kaisu_1", IsPrimaryKey = false, Comment = "着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 31, Length = 6, Name = "chaku_kaisu_2", IsPrimaryKey = false, Comment = "着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 37, Length = 6, Name = "chaku_kaisu_3", IsPrimaryKey = false, Comment = "着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 43, Length = 6, Name = "chaku_kaisu_4", IsPrimaryKey = false, Comment = "着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 49, Length = 6, Name = "chaku_kaisu_5", IsPrimaryKey = false, Comment = "着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 55, Length = 6, Name = "chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "着回数着外", DataType = "CHAR" }
                                }
                            }
                        }
                    }
                }
            };
            recordDefinitions.Add(bnRecord);

            // HN - 繁殖馬マスタ
            var hnRecord = new RecordDefinition
            {
                RecordTypeId = "HN",
                CreationDateField = new FieldDefinition { Position = 4, Length = 8 },
                Table = new TableDefinition
                {
                    Name = "hanshoku_uma_master",
                    Comment = "繁殖馬マスタ",
                    Fields = new List<FieldDefinition>
                    {
                        new NormalFieldDefinition { Position = 1, Length = 2, Name = "record_type_id", IsPrimaryKey = false, Comment = "レコード種別ID", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 3, Length = 1, Name = "data_kubun", IsPrimaryKey = false, Comment = "データ区分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 12, Length = 10, Name = "hanshoku_toroku_number", IsPrimaryKey = true, Comment = "繁殖登録番号", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 22, Length = 8, Name = "yobi_1", IsPrimaryKey = false, Comment = "予備", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 30, Length = 10, Name = "ketto_toroku_number", IsPrimaryKey = false, Comment = "血統登録番号", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 40, Length = 1, Name = "yobi_2", IsPrimaryKey = false, Comment = "予備", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 41, Length = 36, Name = "uma_name", IsPrimaryKey = false, Comment = "馬名", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 77, Length = 40, Name = "uma_name_hankaku_kana", IsPrimaryKey = false, Comment = "馬名半角カナ", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 117, Length = 80, Name = "uma_name_eng", IsPrimaryKey = false, Comment = "馬名欧字", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 197, Length = 4, Name = "birth_year", IsPrimaryKey = false, Comment = "生年", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 201, Length = 1, Name = "seibetsu_code", IsPrimaryKey = false, Comment = "性別コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 202, Length = 1, Name = "hinshu_code", IsPrimaryKey = false, Comment = "品種コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 203, Length = 2, Name = "keiro_code", IsPrimaryKey = false, Comment = "毛色コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 205, Length = 1, Name = "hanshokuba_mochikomi_kubun", IsPrimaryKey = false, Comment = "繁殖馬持込区分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 206, Length = 4, Name = "yunyu_year", IsPrimaryKey = false, Comment = "輸入年", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 210, Length = 20, Name = "sanchi_name", IsPrimaryKey = false, Comment = "産地名", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 230, Length = 10, Name = "chichi_uma_hanshoku_toroku_number", IsPrimaryKey = false, Comment = "父馬繁殖登録番号", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 240, Length = 10, Name = "haha_uma_hanshoku_toroku_number", IsPrimaryKey = false, Comment = "母馬繁殖登録番号", DataType = "CHAR" }
                    }
                }
            };
            recordDefinitions.Add(hnRecord);

            // SK - 産駒マスタ
            var skRecord = new RecordDefinition
            {
                RecordTypeId = "SK",
                CreationDateField = new FieldDefinition { Position = 4, Length = 8 },
                Table = new TableDefinition
                {
                    Name = "sanku_master",
                    Comment = "産駒マスタ",
                    Fields = new List<FieldDefinition>
                    {
                        new NormalFieldDefinition { Position = 1, Length = 2, Name = "record_type_id", IsPrimaryKey = false, Comment = "レコード種別ID", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 3, Length = 1, Name = "data_kubun", IsPrimaryKey = false, Comment = "データ区分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 12, Length = 10, Name = "ketto_toroku_number", IsPrimaryKey = true, Comment = "血統登録番号", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 22, Length = 8, Name = "birth_date", IsPrimaryKey = false, Comment = "生年月日", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 30, Length = 1, Name = "seibetsu_code", IsPrimaryKey = false, Comment = "性別コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 31, Length = 1, Name = "hinshu_code", IsPrimaryKey = false, Comment = "品種コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 32, Length = 2, Name = "keiro_code", IsPrimaryKey = false, Comment = "毛色コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 34, Length = 1, Name = "sanku_mochikomi_kubun", IsPrimaryKey = false, Comment = "産駒持込区分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 35, Length = 4, Name = "yunyuu_year", IsPrimaryKey = false, Comment = "輸入年", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 39, Length = 8, Name = "seisansha_code", IsPrimaryKey = false, Comment = "生産者コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 47, Length = 20, Name = "sanchi_name", IsPrimaryKey = false, Comment = "産地名", DataType = "CHAR" },
                        
                        // 3代血統繁殖登録番号 (14回繰り返し)
                        new NormalFieldDefinition { Position = 67, Length = 10, Name = "chichi_hanshoku_toroku_number", IsPrimaryKey = false, Comment = "父繁殖登録番号", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 77, Length = 10, Name = "haha_hanshoku_toroku_number", IsPrimaryKey = false, Comment = "母繁殖登録番号", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 87, Length = 10, Name = "chichi_chichi_hanshoku_toroku_number", IsPrimaryKey = false, Comment = "父父繁殖登録番号", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 97, Length = 10, Name = "chichi_haha_hanshoku_toroku_number", IsPrimaryKey = false, Comment = "父母繁殖登録番号", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 107, Length = 10, Name = "haha_chichi_hanshoku_toroku_number", IsPrimaryKey = false, Comment = "母父繁殖登録番号", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 117, Length = 10, Name = "haha_haha_hanshoku_toroku_number", IsPrimaryKey = false, Comment = "母母繁殖登録番号", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 127, Length = 10, Name = "chichi_chichi_chichi_hanshoku_toroku_number", IsPrimaryKey = false, Comment = "父父父繁殖登録番号", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 137, Length = 10, Name = "chichi_chichi_haha_hanshoku_toroku_number", IsPrimaryKey = false, Comment = "父父母繁殖登録番号", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 147, Length = 10, Name = "chichi_haha_chichi_hanshoku_toroku_number", IsPrimaryKey = false, Comment = "父母父繁殖登録番号", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 157, Length = 10, Name = "chichi_haha_haha_hanshoku_toroku_number", IsPrimaryKey = false, Comment = "父母母繁殖登録番号", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 167, Length = 10, Name = "haha_chichi_chichi_hanshoku_toroku_number", IsPrimaryKey = false, Comment = "母父父繁殖登録番号", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 177, Length = 10, Name = "haha_chichi_haha_hanshoku_toroku_number", IsPrimaryKey = false, Comment = "母父母繁殖登録番号", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 187, Length = 10, Name = "haha_haha_chichi_hanshoku_toroku_number", IsPrimaryKey = false, Comment = "母母父繁殖登録番号", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 197, Length = 10, Name = "haha_haha_haha_hanshoku_toroku_number", IsPrimaryKey = false, Comment = "母母母繁殖登録番号", DataType = "CHAR" }
                    }
                }
            };
            recordDefinitions.Add(skRecord);

            // CK - 出走別着度数
            var ckRecord = new RecordDefinition
            {
                RecordTypeId = "CK",
                CreationDateField = new FieldDefinition { Position = 4, Length = 8 },
                Table = new TableDefinition
                {
                    Name = "shusso_betsu_chakudosu",
                    Comment = "出走別着度数",
                    Fields = new List<FieldDefinition>
                    {
                        new NormalFieldDefinition { Position = 1, Length = 2, Name = "record_type_id", IsPrimaryKey = false, Comment = "レコード種別ID", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 3, Length = 1, Name = "data_kubun", IsPrimaryKey = false, Comment = "データ区分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 12, Length = 4, Name = "kaisai_year", IsPrimaryKey = true, Comment = "開催年", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 16, Length = 4, Name = "kaisai_monthday", IsPrimaryKey = true, Comment = "開催月日", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 20, Length = 2, Name = "keibajo_code", IsPrimaryKey = true, Comment = "競馬場コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 22, Length = 2, Name = "kaisai_kai", IsPrimaryKey = true, Comment = "開催回", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 24, Length = 2, Name = "kaisai_nichime", IsPrimaryKey = true, Comment = "開催日目", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 26, Length = 2, Name = "race_number", IsPrimaryKey = true, Comment = "レース番号", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 28, Length = 10, Name = "ketto_toroku_number", IsPrimaryKey = true, Comment = "血統登録番号", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 38, Length = 36, Name = "uma_name", IsPrimaryKey = false, Comment = "馬名", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 74, Length = 9, Name = "heichi_hon_shokin_ruikei", IsPrimaryKey = false, Comment = "平地本賞金累計", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 83, Length = 9, Name = "shogai_hon_shokin_ruikei", IsPrimaryKey = false, Comment = "障害本賞金累計", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 92, Length = 9, Name = "heichi_fuka_shokin_ruikei", IsPrimaryKey = false, Comment = "平地付加賞金累計", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 101, Length = 9, Name = "shogai_fuka_shokin_ruikei", IsPrimaryKey = false, Comment = "障害付加賞金累計", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 110, Length = 9, Name = "heichi_shutoku_shokin_ruikei", IsPrimaryKey = false, Comment = "平地収得賞金累計", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 119, Length = 9, Name = "shogai_shutoku_shokin_ruikei", IsPrimaryKey = false, Comment = "障害収得賞金累計", DataType = "CHAR" },
                        
                        // 総合着回数 (6回繰り返し: 1着～5着、着外)
                        new NormalFieldDefinition { Position = 128, Length = 3, Name = "sogo_chaku_kaisu_1", IsPrimaryKey = false, Comment = "総合着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 131, Length = 3, Name = "sogo_chaku_kaisu_2", IsPrimaryKey = false, Comment = "総合着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 134, Length = 3, Name = "sogo_chaku_kaisu_3", IsPrimaryKey = false, Comment = "総合着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 137, Length = 3, Name = "sogo_chaku_kaisu_4", IsPrimaryKey = false, Comment = "総合着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 140, Length = 3, Name = "sogo_chaku_kaisu_5", IsPrimaryKey = false, Comment = "総合着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 143, Length = 3, Name = "sogo_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "総合着回数着外", DataType = "CHAR" },
                        
                        // 中央合計着回数 (6回繰り返し: 1着～5着、着外)
                        new NormalFieldDefinition { Position = 146, Length = 3, Name = "chuo_gokei_chaku_kaisu_1", IsPrimaryKey = false, Comment = "中央合計着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 149, Length = 3, Name = "chuo_gokei_chaku_kaisu_2", IsPrimaryKey = false, Comment = "中央合計着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 152, Length = 3, Name = "chuo_gokei_chaku_kaisu_3", IsPrimaryKey = false, Comment = "中央合計着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 155, Length = 3, Name = "chuo_gokei_chaku_kaisu_4", IsPrimaryKey = false, Comment = "中央合計着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 158, Length = 3, Name = "chuo_gokei_chaku_kaisu_5", IsPrimaryKey = false, Comment = "中央合計着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 161, Length = 3, Name = "chuo_gokei_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "中央合計着回数着外", DataType = "CHAR" },
                        
                        // 馬場別着回数
                        new NormalFieldDefinition { Position = 164, Length = 3, Name = "shiba_choku_chaku_kaisu_1", IsPrimaryKey = false, Comment = "芝直着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 167, Length = 3, Name = "shiba_choku_chaku_kaisu_2", IsPrimaryKey = false, Comment = "芝直着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 170, Length = 3, Name = "shiba_choku_chaku_kaisu_3", IsPrimaryKey = false, Comment = "芝直着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 173, Length = 3, Name = "shiba_choku_chaku_kaisu_4", IsPrimaryKey = false, Comment = "芝直着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 176, Length = 3, Name = "shiba_choku_chaku_kaisu_5", IsPrimaryKey = false, Comment = "芝直着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 179, Length = 3, Name = "shiba_choku_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "芝直着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 182, Length = 3, Name = "shiba_migi_chaku_kaisu_1", IsPrimaryKey = false, Comment = "芝右着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 185, Length = 3, Name = "shiba_migi_chaku_kaisu_2", IsPrimaryKey = false, Comment = "芝右着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 188, Length = 3, Name = "shiba_migi_chaku_kaisu_3", IsPrimaryKey = false, Comment = "芝右着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 191, Length = 3, Name = "shiba_migi_chaku_kaisu_4", IsPrimaryKey = false, Comment = "芝右着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 194, Length = 3, Name = "shiba_migi_chaku_kaisu_5", IsPrimaryKey = false, Comment = "芝右着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 197, Length = 3, Name = "shiba_migi_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "芝右着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 200, Length = 3, Name = "shiba_hidari_chaku_kaisu_1", IsPrimaryKey = false, Comment = "芝左着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 203, Length = 3, Name = "shiba_hidari_chaku_kaisu_2", IsPrimaryKey = false, Comment = "芝左着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 206, Length = 3, Name = "shiba_hidari_chaku_kaisu_3", IsPrimaryKey = false, Comment = "芝左着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 209, Length = 3, Name = "shiba_hidari_chaku_kaisu_4", IsPrimaryKey = false, Comment = "芝左着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 212, Length = 3, Name = "shiba_hidari_chaku_kaisu_5", IsPrimaryKey = false, Comment = "芝左着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 215, Length = 3, Name = "shiba_hidari_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "芝左着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 218, Length = 3, Name = "dirt_choku_chaku_kaisu_1", IsPrimaryKey = false, Comment = "ダ直着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 221, Length = 3, Name = "dirt_choku_chaku_kaisu_2", IsPrimaryKey = false, Comment = "ダ直着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 224, Length = 3, Name = "dirt_choku_chaku_kaisu_3", IsPrimaryKey = false, Comment = "ダ直着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 227, Length = 3, Name = "dirt_choku_chaku_kaisu_4", IsPrimaryKey = false, Comment = "ダ直着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 230, Length = 3, Name = "dirt_choku_chaku_kaisu_5", IsPrimaryKey = false, Comment = "ダ直着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 233, Length = 3, Name = "dirt_choku_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "ダ直着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 236, Length = 3, Name = "dirt_migi_chaku_kaisu_1", IsPrimaryKey = false, Comment = "ダ右着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 239, Length = 3, Name = "dirt_migi_chaku_kaisu_2", IsPrimaryKey = false, Comment = "ダ右着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 242, Length = 3, Name = "dirt_migi_chaku_kaisu_3", IsPrimaryKey = false, Comment = "ダ右着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 245, Length = 3, Name = "dirt_migi_chaku_kaisu_4", IsPrimaryKey = false, Comment = "ダ右着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 248, Length = 3, Name = "dirt_migi_chaku_kaisu_5", IsPrimaryKey = false, Comment = "ダ右着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 251, Length = 3, Name = "dirt_migi_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "ダ右着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 254, Length = 3, Name = "dirt_hidari_chaku_kaisu_1", IsPrimaryKey = false, Comment = "ダ左着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 257, Length = 3, Name = "dirt_hidari_chaku_kaisu_2", IsPrimaryKey = false, Comment = "ダ左着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 260, Length = 3, Name = "dirt_hidari_chaku_kaisu_3", IsPrimaryKey = false, Comment = "ダ左着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 263, Length = 3, Name = "dirt_hidari_chaku_kaisu_4", IsPrimaryKey = false, Comment = "ダ左着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 266, Length = 3, Name = "dirt_hidari_chaku_kaisu_5", IsPrimaryKey = false, Comment = "ダ左着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 269, Length = 3, Name = "dirt_hidari_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "ダ左着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 272, Length = 3, Name = "shogai_chaku_kaisu_1", IsPrimaryKey = false, Comment = "障害着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 275, Length = 3, Name = "shogai_chaku_kaisu_2", IsPrimaryKey = false, Comment = "障害着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 278, Length = 3, Name = "shogai_chaku_kaisu_3", IsPrimaryKey = false, Comment = "障害着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 281, Length = 3, Name = "shogai_chaku_kaisu_4", IsPrimaryKey = false, Comment = "障害着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 284, Length = 3, Name = "shogai_chaku_kaisu_5", IsPrimaryKey = false, Comment = "障害着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 287, Length = 3, Name = "shogai_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "障害着回数着外", DataType = "CHAR" },
                        
                        // 馬場状態別着回数
                        new NormalFieldDefinition { Position = 290, Length = 3, Name = "shiba_ryo_chaku_kaisu_1", IsPrimaryKey = false, Comment = "芝良着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 293, Length = 3, Name = "shiba_ryo_chaku_kaisu_2", IsPrimaryKey = false, Comment = "芝良着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 296, Length = 3, Name = "shiba_ryo_chaku_kaisu_3", IsPrimaryKey = false, Comment = "芝良着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 299, Length = 3, Name = "shiba_ryo_chaku_kaisu_4", IsPrimaryKey = false, Comment = "芝良着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 302, Length = 3, Name = "shiba_ryo_chaku_kaisu_5", IsPrimaryKey = false, Comment = "芝良着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 305, Length = 3, Name = "shiba_ryo_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "芝良着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 308, Length = 3, Name = "shiba_yayaomo_chaku_kaisu_1", IsPrimaryKey = false, Comment = "芝稍着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 311, Length = 3, Name = "shiba_yayaomo_chaku_kaisu_2", IsPrimaryKey = false, Comment = "芝稍着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 314, Length = 3, Name = "shiba_yayaomo_chaku_kaisu_3", IsPrimaryKey = false, Comment = "芝稍着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 317, Length = 3, Name = "shiba_yayaomo_chaku_kaisu_4", IsPrimaryKey = false, Comment = "芝稍着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 320, Length = 3, Name = "shiba_yayaomo_chaku_kaisu_5", IsPrimaryKey = false, Comment = "芝稍着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 323, Length = 3, Name = "shiba_yayaomo_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "芝稍着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 326, Length = 3, Name = "shiba_omo_chaku_kaisu_1", IsPrimaryKey = false, Comment = "芝重着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 329, Length = 3, Name = "shiba_omo_chaku_kaisu_2", IsPrimaryKey = false, Comment = "芝重着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 332, Length = 3, Name = "shiba_omo_chaku_kaisu_3", IsPrimaryKey = false, Comment = "芝重着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 335, Length = 3, Name = "shiba_omo_chaku_kaisu_4", IsPrimaryKey = false, Comment = "芝重着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 338, Length = 3, Name = "shiba_omo_chaku_kaisu_5", IsPrimaryKey = false, Comment = "芝重着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 341, Length = 3, Name = "shiba_omo_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "芝重着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 344, Length = 3, Name = "shiba_furyo_chaku_kaisu_1", IsPrimaryKey = false, Comment = "芝不着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 347, Length = 3, Name = "shiba_furyo_chaku_kaisu_2", IsPrimaryKey = false, Comment = "芝不着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 350, Length = 3, Name = "shiba_furyo_chaku_kaisu_3", IsPrimaryKey = false, Comment = "芝不着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 353, Length = 3, Name = "shiba_furyo_chaku_kaisu_4", IsPrimaryKey = false, Comment = "芝不着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 356, Length = 3, Name = "shiba_furyo_chaku_kaisu_5", IsPrimaryKey = false, Comment = "芝不着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 359, Length = 3, Name = "shiba_furyo_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "芝不着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 362, Length = 3, Name = "dirt_ryo_chaku_kaisu_1", IsPrimaryKey = false, Comment = "ダ良着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 365, Length = 3, Name = "dirt_ryo_chaku_kaisu_2", IsPrimaryKey = false, Comment = "ダ良着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 368, Length = 3, Name = "dirt_ryo_chaku_kaisu_3", IsPrimaryKey = false, Comment = "ダ良着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 371, Length = 3, Name = "dirt_ryo_chaku_kaisu_4", IsPrimaryKey = false, Comment = "ダ良着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 374, Length = 3, Name = "dirt_ryo_chaku_kaisu_5", IsPrimaryKey = false, Comment = "ダ良着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 377, Length = 3, Name = "dirt_ryo_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "ダ良着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 380, Length = 3, Name = "dirt_yayaomo_chaku_kaisu_1", IsPrimaryKey = false, Comment = "ダ稍着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 383, Length = 3, Name = "dirt_yayaomo_chaku_kaisu_2", IsPrimaryKey = false, Comment = "ダ稍着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 386, Length = 3, Name = "dirt_yayaomo_chaku_kaisu_3", IsPrimaryKey = false, Comment = "ダ稍着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 389, Length = 3, Name = "dirt_yayaomo_chaku_kaisu_4", IsPrimaryKey = false, Comment = "ダ稍着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 392, Length = 3, Name = "dirt_yayaomo_chaku_kaisu_5", IsPrimaryKey = false, Comment = "ダ稍着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 395, Length = 3, Name = "dirt_yayaomo_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "ダ稍着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 398, Length = 3, Name = "dirt_omo_chaku_kaisu_1", IsPrimaryKey = false, Comment = "ダ重着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 401, Length = 3, Name = "dirt_omo_chaku_kaisu_2", IsPrimaryKey = false, Comment = "ダ重着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 404, Length = 3, Name = "dirt_omo_chaku_kaisu_3", IsPrimaryKey = false, Comment = "ダ重着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 407, Length = 3, Name = "dirt_omo_chaku_kaisu_4", IsPrimaryKey = false, Comment = "ダ重着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 410, Length = 3, Name = "dirt_omo_chaku_kaisu_5", IsPrimaryKey = false, Comment = "ダ重着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 413, Length = 3, Name = "dirt_omo_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "ダ重着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 416, Length = 3, Name = "dirt_furyo_chaku_kaisu_1", IsPrimaryKey = false, Comment = "ダ不着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 419, Length = 3, Name = "dirt_furyo_chaku_kaisu_2", IsPrimaryKey = false, Comment = "ダ不着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 422, Length = 3, Name = "dirt_furyo_chaku_kaisu_3", IsPrimaryKey = false, Comment = "ダ不着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 425, Length = 3, Name = "dirt_furyo_chaku_kaisu_4", IsPrimaryKey = false, Comment = "ダ不着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 428, Length = 3, Name = "dirt_furyo_chaku_kaisu_5", IsPrimaryKey = false, Comment = "ダ不着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 431, Length = 3, Name = "dirt_furyo_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "ダ不着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 434, Length = 3, Name = "shogai_ryo_chaku_kaisu_1", IsPrimaryKey = false, Comment = "障良着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 437, Length = 3, Name = "shogai_ryo_chaku_kaisu_2", IsPrimaryKey = false, Comment = "障良着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 440, Length = 3, Name = "shogai_ryo_chaku_kaisu_3", IsPrimaryKey = false, Comment = "障良着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 443, Length = 3, Name = "shogai_ryo_chaku_kaisu_4", IsPrimaryKey = false, Comment = "障良着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 446, Length = 3, Name = "shogai_ryo_chaku_kaisu_5", IsPrimaryKey = false, Comment = "障良着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 449, Length = 3, Name = "shogai_ryo_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "障良着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 452, Length = 3, Name = "shogai_yayaomo_chaku_kaisu_1", IsPrimaryKey = false, Comment = "障稍着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 455, Length = 3, Name = "shogai_yayaomo_chaku_kaisu_2", IsPrimaryKey = false, Comment = "障稍着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 458, Length = 3, Name = "shogai_yayaomo_chaku_kaisu_3", IsPrimaryKey = false, Comment = "障稍着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 461, Length = 3, Name = "shogai_yayaomo_chaku_kaisu_4", IsPrimaryKey = false, Comment = "障稍着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 464, Length = 3, Name = "shogai_yayaomo_chaku_kaisu_5", IsPrimaryKey = false, Comment = "障稍着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 467, Length = 3, Name = "shogai_yayaomo_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "障稍着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 470, Length = 3, Name = "shogai_omo_chaku_kaisu_1", IsPrimaryKey = false, Comment = "障重着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 473, Length = 3, Name = "shogai_omo_chaku_kaisu_2", IsPrimaryKey = false, Comment = "障重着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 476, Length = 3, Name = "shogai_omo_chaku_kaisu_3", IsPrimaryKey = false, Comment = "障重着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 479, Length = 3, Name = "shogai_omo_chaku_kaisu_4", IsPrimaryKey = false, Comment = "障重着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 482, Length = 3, Name = "shogai_omo_chaku_kaisu_5", IsPrimaryKey = false, Comment = "障重着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 485, Length = 3, Name = "shogai_omo_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "障重着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 488, Length = 3, Name = "shogai_furyo_chaku_kaisu_1", IsPrimaryKey = false, Comment = "障不着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 491, Length = 3, Name = "shogai_furyo_chaku_kaisu_2", IsPrimaryKey = false, Comment = "障不着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 494, Length = 3, Name = "shogai_furyo_chaku_kaisu_3", IsPrimaryKey = false, Comment = "障不着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 497, Length = 3, Name = "shogai_furyo_chaku_kaisu_4", IsPrimaryKey = false, Comment = "障不着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 500, Length = 3, Name = "shogai_furyo_chaku_kaisu_5", IsPrimaryKey = false, Comment = "障不着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 503, Length = 3, Name = "shogai_furyo_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "障不着回数着外", DataType = "CHAR" },
                        
                        // 距離別着回数
                        new NormalFieldDefinition { Position = 506, Length = 3, Name = "shiba_1200_ika_chaku_kaisu_1", IsPrimaryKey = false, Comment = "芝1200以下着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 509, Length = 3, Name = "shiba_1200_ika_chaku_kaisu_2", IsPrimaryKey = false, Comment = "芝1200以下着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 512, Length = 3, Name = "shiba_1200_ika_chaku_kaisu_3", IsPrimaryKey = false, Comment = "芝1200以下着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 515, Length = 3, Name = "shiba_1200_ika_chaku_kaisu_4", IsPrimaryKey = false, Comment = "芝1200以下着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 518, Length = 3, Name = "shiba_1200_ika_chaku_kaisu_5", IsPrimaryKey = false, Comment = "芝1200以下着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 521, Length = 3, Name = "shiba_1200_ika_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "芝1200以下着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 524, Length = 3, Name = "shiba_1201_1400_chaku_kaisu_1", IsPrimaryKey = false, Comment = "芝1201-1400着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 527, Length = 3, Name = "shiba_1201_1400_chaku_kaisu_2", IsPrimaryKey = false, Comment = "芝1201-1400着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 530, Length = 3, Name = "shiba_1201_1400_chaku_kaisu_3", IsPrimaryKey = false, Comment = "芝1201-1400着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 533, Length = 3, Name = "shiba_1201_1400_chaku_kaisu_4", IsPrimaryKey = false, Comment = "芝1201-1400着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 536, Length = 3, Name = "shiba_1201_1400_chaku_kaisu_5", IsPrimaryKey = false, Comment = "芝1201-1400着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 539, Length = 3, Name = "shiba_1201_1400_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "芝1201-1400着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 542, Length = 3, Name = "shiba_1401_1600_chaku_kaisu_1", IsPrimaryKey = false, Comment = "芝1401-1600着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 545, Length = 3, Name = "shiba_1401_1600_chaku_kaisu_2", IsPrimaryKey = false, Comment = "芝1401-1600着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 548, Length = 3, Name = "shiba_1401_1600_chaku_kaisu_3", IsPrimaryKey = false, Comment = "芝1401-1600着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 551, Length = 3, Name = "shiba_1401_1600_chaku_kaisu_4", IsPrimaryKey = false, Comment = "芝1401-1600着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 554, Length = 3, Name = "shiba_1401_1600_chaku_kaisu_5", IsPrimaryKey = false, Comment = "芝1401-1600着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 557, Length = 3, Name = "shiba_1401_1600_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "芝1401-1600着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 560, Length = 3, Name = "shiba_1601_1800_chaku_kaisu_1", IsPrimaryKey = false, Comment = "芝1601-1800着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 563, Length = 3, Name = "shiba_1601_1800_chaku_kaisu_2", IsPrimaryKey = false, Comment = "芝1601-1800着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 566, Length = 3, Name = "shiba_1601_1800_chaku_kaisu_3", IsPrimaryKey = false, Comment = "芝1601-1800着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 569, Length = 3, Name = "shiba_1601_1800_chaku_kaisu_4", IsPrimaryKey = false, Comment = "芝1601-1800着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 572, Length = 3, Name = "shiba_1601_1800_chaku_kaisu_5", IsPrimaryKey = false, Comment = "芝1601-1800着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 575, Length = 3, Name = "shiba_1601_1800_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "芝1601-1800着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 578, Length = 3, Name = "shiba_1801_2000_chaku_kaisu_1", IsPrimaryKey = false, Comment = "芝1801-2000着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 581, Length = 3, Name = "shiba_1801_2000_chaku_kaisu_2", IsPrimaryKey = false, Comment = "芝1801-2000着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 584, Length = 3, Name = "shiba_1801_2000_chaku_kaisu_3", IsPrimaryKey = false, Comment = "芝1801-2000着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 587, Length = 3, Name = "shiba_1801_2000_chaku_kaisu_4", IsPrimaryKey = false, Comment = "芝1801-2000着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 590, Length = 3, Name = "shiba_1801_2000_chaku_kaisu_5", IsPrimaryKey = false, Comment = "芝1801-2000着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 593, Length = 3, Name = "shiba_1801_2000_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "芝1801-2000着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 596, Length = 3, Name = "shiba_2001_2200_chaku_kaisu_1", IsPrimaryKey = false, Comment = "芝2001-2200着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 599, Length = 3, Name = "shiba_2001_2200_chaku_kaisu_2", IsPrimaryKey = false, Comment = "芝2001-2200着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 602, Length = 3, Name = "shiba_2001_2200_chaku_kaisu_3", IsPrimaryKey = false, Comment = "芝2001-2200着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 605, Length = 3, Name = "shiba_2001_2200_chaku_kaisu_4", IsPrimaryKey = false, Comment = "芝2001-2200着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 608, Length = 3, Name = "shiba_2001_2200_chaku_kaisu_5", IsPrimaryKey = false, Comment = "芝2001-2200着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 611, Length = 3, Name = "shiba_2001_2200_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "芝2001-2200着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 614, Length = 3, Name = "shiba_2201_2400_chaku_kaisu_1", IsPrimaryKey = false, Comment = "芝2201-2400着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 617, Length = 3, Name = "shiba_2201_2400_chaku_kaisu_2", IsPrimaryKey = false, Comment = "芝2201-2400着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 620, Length = 3, Name = "shiba_2201_2400_chaku_kaisu_3", IsPrimaryKey = false, Comment = "芝2201-2400着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 623, Length = 3, Name = "shiba_2201_2400_chaku_kaisu_4", IsPrimaryKey = false, Comment = "芝2201-2400着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 626, Length = 3, Name = "shiba_2201_2400_chaku_kaisu_5", IsPrimaryKey = false, Comment = "芝2201-2400着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 629, Length = 3, Name = "shiba_2201_2400_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "芝2201-2400着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 632, Length = 3, Name = "shiba_2401_2800_chaku_kaisu_1", IsPrimaryKey = false, Comment = "芝2401-2800着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 635, Length = 3, Name = "shiba_2401_2800_chaku_kaisu_2", IsPrimaryKey = false, Comment = "芝2401-2800着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 638, Length = 3, Name = "shiba_2401_2800_chaku_kaisu_3", IsPrimaryKey = false, Comment = "芝2401-2800着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 641, Length = 3, Name = "shiba_2401_2800_chaku_kaisu_4", IsPrimaryKey = false, Comment = "芝2401-2800着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 644, Length = 3, Name = "shiba_2401_2800_chaku_kaisu_5", IsPrimaryKey = false, Comment = "芝2401-2800着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 647, Length = 3, Name = "shiba_2401_2800_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "芝2401-2800着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 650, Length = 3, Name = "shiba_2801_ijo_chaku_kaisu_1", IsPrimaryKey = false, Comment = "芝2801以上着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 653, Length = 3, Name = "shiba_2801_ijo_chaku_kaisu_2", IsPrimaryKey = false, Comment = "芝2801以上着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 656, Length = 3, Name = "shiba_2801_ijo_chaku_kaisu_3", IsPrimaryKey = false, Comment = "芝2801以上着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 659, Length = 3, Name = "shiba_2801_ijo_chaku_kaisu_4", IsPrimaryKey = false, Comment = "芝2801以上着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 662, Length = 3, Name = "shiba_2801_ijo_chaku_kaisu_5", IsPrimaryKey = false, Comment = "芝2801以上着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 665, Length = 3, Name = "shiba_2801_ijo_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "芝2801以上着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 668, Length = 3, Name = "dirt_1200_ika_chaku_kaisu_1", IsPrimaryKey = false, Comment = "ダ1200以下着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 671, Length = 3, Name = "dirt_1200_ika_chaku_kaisu_2", IsPrimaryKey = false, Comment = "ダ1200以下着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 674, Length = 3, Name = "dirt_1200_ika_chaku_kaisu_3", IsPrimaryKey = false, Comment = "ダ1200以下着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 677, Length = 3, Name = "dirt_1200_ika_chaku_kaisu_4", IsPrimaryKey = false, Comment = "ダ1200以下着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 680, Length = 3, Name = "dirt_1200_ika_chaku_kaisu_5", IsPrimaryKey = false, Comment = "ダ1200以下着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 683, Length = 3, Name = "dirt_1200_ika_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "ダ1200以下着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 686, Length = 3, Name = "dirt_1201_1400_chaku_kaisu_1", IsPrimaryKey = false, Comment = "ダ1201-1400着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 689, Length = 3, Name = "dirt_1201_1400_chaku_kaisu_2", IsPrimaryKey = false, Comment = "ダ1201-1400着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 692, Length = 3, Name = "dirt_1201_1400_chaku_kaisu_3", IsPrimaryKey = false, Comment = "ダ1201-1400着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 695, Length = 3, Name = "dirt_1201_1400_chaku_kaisu_4", IsPrimaryKey = false, Comment = "ダ1201-1400着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 698, Length = 3, Name = "dirt_1201_1400_chaku_kaisu_5", IsPrimaryKey = false, Comment = "ダ1201-1400着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 701, Length = 3, Name = "dirt_1201_1400_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "ダ1201-1400着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 704, Length = 3, Name = "dirt_1401_1600_chaku_kaisu_1", IsPrimaryKey = false, Comment = "ダ1401-1600着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 707, Length = 3, Name = "dirt_1401_1600_chaku_kaisu_2", IsPrimaryKey = false, Comment = "ダ1401-1600着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 710, Length = 3, Name = "dirt_1401_1600_chaku_kaisu_3", IsPrimaryKey = false, Comment = "ダ1401-1600着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 713, Length = 3, Name = "dirt_1401_1600_chaku_kaisu_4", IsPrimaryKey = false, Comment = "ダ1401-1600着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 716, Length = 3, Name = "dirt_1401_1600_chaku_kaisu_5", IsPrimaryKey = false, Comment = "ダ1401-1600着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 719, Length = 3, Name = "dirt_1401_1600_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "ダ1401-1600着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 722, Length = 3, Name = "dirt_1601_1800_chaku_kaisu_1", IsPrimaryKey = false, Comment = "ダ1601-1800着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 725, Length = 3, Name = "dirt_1601_1800_chaku_kaisu_2", IsPrimaryKey = false, Comment = "ダ1601-1800着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 728, Length = 3, Name = "dirt_1601_1800_chaku_kaisu_3", IsPrimaryKey = false, Comment = "ダ1601-1800着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 731, Length = 3, Name = "dirt_1601_1800_chaku_kaisu_4", IsPrimaryKey = false, Comment = "ダ1601-1800着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 734, Length = 3, Name = "dirt_1601_1800_chaku_kaisu_5", IsPrimaryKey = false, Comment = "ダ1601-1800着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 737, Length = 3, Name = "dirt_1601_1800_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "ダ1601-1800着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 740, Length = 3, Name = "dirt_1801_2000_chaku_kaisu_1", IsPrimaryKey = false, Comment = "ダ1801-2000着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 743, Length = 3, Name = "dirt_1801_2000_chaku_kaisu_2", IsPrimaryKey = false, Comment = "ダ1801-2000着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 746, Length = 3, Name = "dirt_1801_2000_chaku_kaisu_3", IsPrimaryKey = false, Comment = "ダ1801-2000着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 749, Length = 3, Name = "dirt_1801_2000_chaku_kaisu_4", IsPrimaryKey = false, Comment = "ダ1801-2000着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 752, Length = 3, Name = "dirt_1801_2000_chaku_kaisu_5", IsPrimaryKey = false, Comment = "ダ1801-2000着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 755, Length = 3, Name = "dirt_1801_2000_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "ダ1801-2000着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 758, Length = 3, Name = "dirt_2001_2200_chaku_kaisu_1", IsPrimaryKey = false, Comment = "ダ2001-2200着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 761, Length = 3, Name = "dirt_2001_2200_chaku_kaisu_2", IsPrimaryKey = false, Comment = "ダ2001-2200着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 764, Length = 3, Name = "dirt_2001_2200_chaku_kaisu_3", IsPrimaryKey = false, Comment = "ダ2001-2200着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 767, Length = 3, Name = "dirt_2001_2200_chaku_kaisu_4", IsPrimaryKey = false, Comment = "ダ2001-2200着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 770, Length = 3, Name = "dirt_2001_2200_chaku_kaisu_5", IsPrimaryKey = false, Comment = "ダ2001-2200着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 773, Length = 3, Name = "dirt_2001_2200_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "ダ2001-2200着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 776, Length = 3, Name = "dirt_2201_2400_chaku_kaisu_1", IsPrimaryKey = false, Comment = "ダ2201-2400着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 779, Length = 3, Name = "dirt_2201_2400_chaku_kaisu_2", IsPrimaryKey = false, Comment = "ダ2201-2400着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 782, Length = 3, Name = "dirt_2201_2400_chaku_kaisu_3", IsPrimaryKey = false, Comment = "ダ2201-2400着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 785, Length = 3, Name = "dirt_2201_2400_chaku_kaisu_4", IsPrimaryKey = false, Comment = "ダ2201-2400着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 788, Length = 3, Name = "dirt_2201_2400_chaku_kaisu_5", IsPrimaryKey = false, Comment = "ダ2201-2400着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 791, Length = 3, Name = "dirt_2201_2400_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "ダ2201-2400着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 794, Length = 3, Name = "dirt_2401_2800_chaku_kaisu_1", IsPrimaryKey = false, Comment = "ダ2401-2800着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 797, Length = 3, Name = "dirt_2401_2800_chaku_kaisu_2", IsPrimaryKey = false, Comment = "ダ2401-2800着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 800, Length = 3, Name = "dirt_2401_2800_chaku_kaisu_3", IsPrimaryKey = false, Comment = "ダ2401-2800着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 803, Length = 3, Name = "dirt_2401_2800_chaku_kaisu_4", IsPrimaryKey = false, Comment = "ダ2401-2800着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 806, Length = 3, Name = "dirt_2401_2800_chaku_kaisu_5", IsPrimaryKey = false, Comment = "ダ2401-2800着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 809, Length = 3, Name = "dirt_2401_2800_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "ダ2401-2800着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 812, Length = 3, Name = "dirt_2801_ijo_chaku_kaisu_1", IsPrimaryKey = false, Comment = "ダ2801以上着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 815, Length = 3, Name = "dirt_2801_ijo_chaku_kaisu_2", IsPrimaryKey = false, Comment = "ダ2801以上着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 818, Length = 3, Name = "dirt_2801_ijo_chaku_kaisu_3", IsPrimaryKey = false, Comment = "ダ2801以上着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 821, Length = 3, Name = "dirt_2801_ijo_chaku_kaisu_4", IsPrimaryKey = false, Comment = "ダ2801以上着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 824, Length = 3, Name = "dirt_2801_ijo_chaku_kaisu_5", IsPrimaryKey = false, Comment = "ダ2801以上着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 827, Length = 3, Name = "dirt_2801_ijo_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "ダ2801以上着回数着外", DataType = "CHAR" },
                        
                        // 競馬場別着回数
                        new NormalFieldDefinition { Position = 830, Length = 3, Name = "sapporo_shiba_chaku_kaisu_1", IsPrimaryKey = false, Comment = "札幌芝着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 833, Length = 3, Name = "sapporo_shiba_chaku_kaisu_2", IsPrimaryKey = false, Comment = "札幌芝着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 836, Length = 3, Name = "sapporo_shiba_chaku_kaisu_3", IsPrimaryKey = false, Comment = "札幌芝着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 839, Length = 3, Name = "sapporo_shiba_chaku_kaisu_4", IsPrimaryKey = false, Comment = "札幌芝着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 842, Length = 3, Name = "sapporo_shiba_chaku_kaisu_5", IsPrimaryKey = false, Comment = "札幌芝着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 845, Length = 3, Name = "sapporo_shiba_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "札幌芝着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 848, Length = 3, Name = "hakodate_shiba_chaku_kaisu_1", IsPrimaryKey = false, Comment = "函館芝着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 851, Length = 3, Name = "hakodate_shiba_chaku_kaisu_2", IsPrimaryKey = false, Comment = "函館芝着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 854, Length = 3, Name = "hakodate_shiba_chaku_kaisu_3", IsPrimaryKey = false, Comment = "函館芝着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 857, Length = 3, Name = "hakodate_shiba_chaku_kaisu_4", IsPrimaryKey = false, Comment = "函館芝着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 860, Length = 3, Name = "hakodate_shiba_chaku_kaisu_5", IsPrimaryKey = false, Comment = "函館芝着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 863, Length = 3, Name = "hakodate_shiba_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "函館芝着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 866, Length = 3, Name = "fukushima_shiba_chaku_kaisu_1", IsPrimaryKey = false, Comment = "福島芝着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 869, Length = 3, Name = "fukushima_shiba_chaku_kaisu_2", IsPrimaryKey = false, Comment = "福島芝着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 872, Length = 3, Name = "fukushima_shiba_chaku_kaisu_3", IsPrimaryKey = false, Comment = "福島芝着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 875, Length = 3, Name = "fukushima_shiba_chaku_kaisu_4", IsPrimaryKey = false, Comment = "福島芝着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 878, Length = 3, Name = "fukushima_shiba_chaku_kaisu_5", IsPrimaryKey = false, Comment = "福島芝着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 881, Length = 3, Name = "fukushima_shiba_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "福島芝着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 884, Length = 3, Name = "nigata_shiba_chaku_kaisu_1", IsPrimaryKey = false, Comment = "新潟芝着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 887, Length = 3, Name = "nigata_shiba_chaku_kaisu_2", IsPrimaryKey = false, Comment = "新潟芝着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 890, Length = 3, Name = "nigata_shiba_chaku_kaisu_3", IsPrimaryKey = false, Comment = "新潟芝着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 893, Length = 3, Name = "nigata_shiba_chaku_kaisu_4", IsPrimaryKey = false, Comment = "新潟芝着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 896, Length = 3, Name = "nigata_shiba_chaku_kaisu_5", IsPrimaryKey = false, Comment = "新潟芝着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 899, Length = 3, Name = "nigata_shiba_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "新潟芝着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 902, Length = 3, Name = "tokyo_shiba_chaku_kaisu_1", IsPrimaryKey = false, Comment = "東京芝着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 905, Length = 3, Name = "tokyo_shiba_chaku_kaisu_2", IsPrimaryKey = false, Comment = "東京芝着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 908, Length = 3, Name = "tokyo_shiba_chaku_kaisu_3", IsPrimaryKey = false, Comment = "東京芝着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 911, Length = 3, Name = "tokyo_shiba_chaku_kaisu_4", IsPrimaryKey = false, Comment = "東京芝着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 914, Length = 3, Name = "tokyo_shiba_chaku_kaisu_5", IsPrimaryKey = false, Comment = "東京芝着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 917, Length = 3, Name = "tokyo_shiba_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "東京芝着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 920, Length = 3, Name = "nakayama_shiba_chaku_kaisu_1", IsPrimaryKey = false, Comment = "中山芝着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 923, Length = 3, Name = "nakayama_shiba_chaku_kaisu_2", IsPrimaryKey = false, Comment = "中山芝着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 926, Length = 3, Name = "nakayama_shiba_chaku_kaisu_3", IsPrimaryKey = false, Comment = "中山芝着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 929, Length = 3, Name = "nakayama_shiba_chaku_kaisu_4", IsPrimaryKey = false, Comment = "中山芝着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 932, Length = 3, Name = "nakayama_shiba_chaku_kaisu_5", IsPrimaryKey = false, Comment = "中山芝着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 935, Length = 3, Name = "nakayama_shiba_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "中山芝着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 938, Length = 3, Name = "chukyo_shiba_chaku_kaisu_1", IsPrimaryKey = false, Comment = "中京芝着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 941, Length = 3, Name = "chukyo_shiba_chaku_kaisu_2", IsPrimaryKey = false, Comment = "中京芝着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 944, Length = 3, Name = "chukyo_shiba_chaku_kaisu_3", IsPrimaryKey = false, Comment = "中京芝着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 947, Length = 3, Name = "chukyo_shiba_chaku_kaisu_4", IsPrimaryKey = false, Comment = "中京芝着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 950, Length = 3, Name = "chukyo_shiba_chaku_kaisu_5", IsPrimaryKey = false, Comment = "中京芝着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 953, Length = 3, Name = "chukyo_shiba_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "中京芝着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 956, Length = 3, Name = "kyoto_shiba_chaku_kaisu_1", IsPrimaryKey = false, Comment = "京都芝着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 959, Length = 3, Name = "kyoto_shiba_chaku_kaisu_2", IsPrimaryKey = false, Comment = "京都芝着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 962, Length = 3, Name = "kyoto_shiba_chaku_kaisu_3", IsPrimaryKey = false, Comment = "京都芝着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 965, Length = 3, Name = "kyoto_shiba_chaku_kaisu_4", IsPrimaryKey = false, Comment = "京都芝着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 968, Length = 3, Name = "kyoto_shiba_chaku_kaisu_5", IsPrimaryKey = false, Comment = "京都芝着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 971, Length = 3, Name = "kyoto_shiba_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "京都芝着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 974, Length = 3, Name = "hanshin_shiba_chaku_kaisu_1", IsPrimaryKey = false, Comment = "阪神芝着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 977, Length = 3, Name = "hanshin_shiba_chaku_kaisu_2", IsPrimaryKey = false, Comment = "阪神芝着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 980, Length = 3, Name = "hanshin_shiba_chaku_kaisu_3", IsPrimaryKey = false, Comment = "阪神芝着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 983, Length = 3, Name = "hanshin_shiba_chaku_kaisu_4", IsPrimaryKey = false, Comment = "阪神芝着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 986, Length = 3, Name = "hanshin_shiba_chaku_kaisu_5", IsPrimaryKey = false, Comment = "阪神芝着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 989, Length = 3, Name = "hanshin_shiba_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "阪神芝着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 992, Length = 3, Name = "kokura_shiba_chaku_kaisu_1", IsPrimaryKey = false, Comment = "小倉芝着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 995, Length = 3, Name = "kokura_shiba_chaku_kaisu_2", IsPrimaryKey = false, Comment = "小倉芝着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 998, Length = 3, Name = "kokura_shiba_chaku_kaisu_3", IsPrimaryKey = false, Comment = "小倉芝着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1001, Length = 3, Name = "kokura_shiba_chaku_kaisu_4", IsPrimaryKey = false, Comment = "小倉芝着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1004, Length = 3, Name = "kokura_shiba_chaku_kaisu_5", IsPrimaryKey = false, Comment = "小倉芝着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1007, Length = 3, Name = "kokura_shiba_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "小倉芝着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1010, Length = 3, Name = "sapporo_dirt_chaku_kaisu_1", IsPrimaryKey = false, Comment = "札幌ダ着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1013, Length = 3, Name = "sapporo_dirt_chaku_kaisu_2", IsPrimaryKey = false, Comment = "札幌ダ着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1016, Length = 3, Name = "sapporo_dirt_chaku_kaisu_3", IsPrimaryKey = false, Comment = "札幌ダ着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1019, Length = 3, Name = "sapporo_dirt_chaku_kaisu_4", IsPrimaryKey = false, Comment = "札幌ダ着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1022, Length = 3, Name = "sapporo_dirt_chaku_kaisu_5", IsPrimaryKey = false, Comment = "札幌ダ着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1025, Length = 3, Name = "sapporo_dirt_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "札幌ダ着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1028, Length = 3, Name = "hakodate_dirt_chaku_kaisu_1", IsPrimaryKey = false, Comment = "函館ダ着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1031, Length = 3, Name = "hakodate_dirt_chaku_kaisu_2", IsPrimaryKey = false, Comment = "函館ダ着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1034, Length = 3, Name = "hakodate_dirt_chaku_kaisu_3", IsPrimaryKey = false, Comment = "函館ダ着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1037, Length = 3, Name = "hakodate_dirt_chaku_kaisu_4", IsPrimaryKey = false, Comment = "函館ダ着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1040, Length = 3, Name = "hakodate_dirt_chaku_kaisu_5", IsPrimaryKey = false, Comment = "函館ダ着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1043, Length = 3, Name = "hakodate_dirt_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "函館ダ着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1046, Length = 3, Name = "fukushima_dirt_chaku_kaisu_1", IsPrimaryKey = false, Comment = "福島ダ着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1049, Length = 3, Name = "fukushima_dirt_chaku_kaisu_2", IsPrimaryKey = false, Comment = "福島ダ着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1052, Length = 3, Name = "fukushima_dirt_chaku_kaisu_3", IsPrimaryKey = false, Comment = "福島ダ着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1055, Length = 3, Name = "fukushima_dirt_chaku_kaisu_4", IsPrimaryKey = false, Comment = "福島ダ着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1058, Length = 3, Name = "fukushima_dirt_chaku_kaisu_5", IsPrimaryKey = false, Comment = "福島ダ着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1061, Length = 3, Name = "fukushima_dirt_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "福島ダ着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1064, Length = 3, Name = "nigata_dirt_chaku_kaisu_1", IsPrimaryKey = false, Comment = "新潟ダ着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1067, Length = 3, Name = "nigata_dirt_chaku_kaisu_2", IsPrimaryKey = false, Comment = "新潟ダ着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1070, Length = 3, Name = "nigata_dirt_chaku_kaisu_3", IsPrimaryKey = false, Comment = "新潟ダ着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1073, Length = 3, Name = "nigata_dirt_chaku_kaisu_4", IsPrimaryKey = false, Comment = "新潟ダ着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1076, Length = 3, Name = "nigata_dirt_chaku_kaisu_5", IsPrimaryKey = false, Comment = "新潟ダ着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1079, Length = 3, Name = "nigata_dirt_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "新潟ダ着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1082, Length = 3, Name = "tokyo_dirt_chaku_kaisu_1", IsPrimaryKey = false, Comment = "東京ダ着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1085, Length = 3, Name = "tokyo_dirt_chaku_kaisu_2", IsPrimaryKey = false, Comment = "東京ダ着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1088, Length = 3, Name = "tokyo_dirt_chaku_kaisu_3", IsPrimaryKey = false, Comment = "東京ダ着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1091, Length = 3, Name = "tokyo_dirt_chaku_kaisu_4", IsPrimaryKey = false, Comment = "東京ダ着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1094, Length = 3, Name = "tokyo_dirt_chaku_kaisu_5", IsPrimaryKey = false, Comment = "東京ダ着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1097, Length = 3, Name = "tokyo_dirt_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "東京ダ着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1100, Length = 3, Name = "nakayama_dirt_chaku_kaisu_1", IsPrimaryKey = false, Comment = "中山ダ着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1103, Length = 3, Name = "nakayama_dirt_chaku_kaisu_2", IsPrimaryKey = false, Comment = "中山ダ着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1106, Length = 3, Name = "nakayama_dirt_chaku_kaisu_3", IsPrimaryKey = false, Comment = "中山ダ着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1109, Length = 3, Name = "nakayama_dirt_chaku_kaisu_4", IsPrimaryKey = false, Comment = "中山ダ着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1112, Length = 3, Name = "nakayama_dirt_chaku_kaisu_5", IsPrimaryKey = false, Comment = "中山ダ着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1115, Length = 3, Name = "nakayama_dirt_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "中山ダ着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1118, Length = 3, Name = "chukyo_dirt_chaku_kaisu_1", IsPrimaryKey = false, Comment = "中京ダ着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1121, Length = 3, Name = "chukyo_dirt_chaku_kaisu_2", IsPrimaryKey = false, Comment = "中京ダ着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1124, Length = 3, Name = "chukyo_dirt_chaku_kaisu_3", IsPrimaryKey = false, Comment = "中京ダ着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1127, Length = 3, Name = "chukyo_dirt_chaku_kaisu_4", IsPrimaryKey = false, Comment = "中京ダ着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1130, Length = 3, Name = "chukyo_dirt_chaku_kaisu_5", IsPrimaryKey = false, Comment = "中京ダ着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1133, Length = 3, Name = "chukyo_dirt_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "中京ダ着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1136, Length = 3, Name = "kyoto_dirt_chaku_kaisu_1", IsPrimaryKey = false, Comment = "京都ダ着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1139, Length = 3, Name = "kyoto_dirt_chaku_kaisu_2", IsPrimaryKey = false, Comment = "京都ダ着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1142, Length = 3, Name = "kyoto_dirt_chaku_kaisu_3", IsPrimaryKey = false, Comment = "京都ダ着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1145, Length = 3, Name = "kyoto_dirt_chaku_kaisu_4", IsPrimaryKey = false, Comment = "京都ダ着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1148, Length = 3, Name = "kyoto_dirt_chaku_kaisu_5", IsPrimaryKey = false, Comment = "京都ダ着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1151, Length = 3, Name = "kyoto_dirt_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "京都ダ着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1154, Length = 3, Name = "hanshin_dirt_chaku_kaisu_1", IsPrimaryKey = false, Comment = "阪神ダ着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1157, Length = 3, Name = "hanshin_dirt_chaku_kaisu_2", IsPrimaryKey = false, Comment = "阪神ダ着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1160, Length = 3, Name = "hanshin_dirt_chaku_kaisu_3", IsPrimaryKey = false, Comment = "阪神ダ着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1163, Length = 3, Name = "hanshin_dirt_chaku_kaisu_4", IsPrimaryKey = false, Comment = "阪神ダ着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1166, Length = 3, Name = "hanshin_dirt_chaku_kaisu_5", IsPrimaryKey = false, Comment = "阪神ダ着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1169, Length = 3, Name = "hanshin_dirt_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "阪神ダ着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1172, Length = 3, Name = "kokura_dirt_chaku_kaisu_1", IsPrimaryKey = false, Comment = "小倉ダ着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1175, Length = 3, Name = "kokura_dirt_chaku_kaisu_2", IsPrimaryKey = false, Comment = "小倉ダ着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1178, Length = 3, Name = "kokura_dirt_chaku_kaisu_3", IsPrimaryKey = false, Comment = "小倉ダ着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1181, Length = 3, Name = "kokura_dirt_chaku_kaisu_4", IsPrimaryKey = false, Comment = "小倉ダ着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1184, Length = 3, Name = "kokura_dirt_chaku_kaisu_5", IsPrimaryKey = false, Comment = "小倉ダ着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1187, Length = 3, Name = "kokura_dirt_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "小倉ダ着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1190, Length = 3, Name = "sapporo_shogai_chaku_kaisu_1", IsPrimaryKey = false, Comment = "札幌障着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1193, Length = 3, Name = "sapporo_shogai_chaku_kaisu_2", IsPrimaryKey = false, Comment = "札幌障着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1196, Length = 3, Name = "sapporo_shogai_chaku_kaisu_3", IsPrimaryKey = false, Comment = "札幌障着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1199, Length = 3, Name = "sapporo_shogai_chaku_kaisu_4", IsPrimaryKey = false, Comment = "札幌障着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1202, Length = 3, Name = "sapporo_shogai_chaku_kaisu_5", IsPrimaryKey = false, Comment = "札幌障着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1205, Length = 3, Name = "sapporo_shogai_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "札幌障着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1208, Length = 3, Name = "hakodate_shogai_chaku_kaisu_1", IsPrimaryKey = false, Comment = "函館障着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1211, Length = 3, Name = "hakodate_shogai_chaku_kaisu_2", IsPrimaryKey = false, Comment = "函館障着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1214, Length = 3, Name = "hakodate_shogai_chaku_kaisu_3", IsPrimaryKey = false, Comment = "函館障着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1217, Length = 3, Name = "hakodate_shogai_chaku_kaisu_4", IsPrimaryKey = false, Comment = "函館障着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1220, Length = 3, Name = "hakodate_shogai_chaku_kaisu_5", IsPrimaryKey = false, Comment = "函館障着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1223, Length = 3, Name = "hakodate_shogai_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "函館障着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1226, Length = 3, Name = "fukushima_shogai_chaku_kaisu_1", IsPrimaryKey = false, Comment = "福島障着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1229, Length = 3, Name = "fukushima_shogai_chaku_kaisu_2", IsPrimaryKey = false, Comment = "福島障着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1232, Length = 3, Name = "fukushima_shogai_chaku_kaisu_3", IsPrimaryKey = false, Comment = "福島障着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1235, Length = 3, Name = "fukushima_shogai_chaku_kaisu_4", IsPrimaryKey = false, Comment = "福島障着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1238, Length = 3, Name = "fukushima_shogai_chaku_kaisu_5", IsPrimaryKey = false, Comment = "福島障着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1241, Length = 3, Name = "fukushima_shogai_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "福島障着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1244, Length = 3, Name = "nigata_shogai_chaku_kaisu_1", IsPrimaryKey = false, Comment = "新潟障着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1247, Length = 3, Name = "nigata_shogai_chaku_kaisu_2", IsPrimaryKey = false, Comment = "新潟障着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1250, Length = 3, Name = "nigata_shogai_chaku_kaisu_3", IsPrimaryKey = false, Comment = "新潟障着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1253, Length = 3, Name = "nigata_shogai_chaku_kaisu_4", IsPrimaryKey = false, Comment = "新潟障着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1256, Length = 3, Name = "nigata_shogai_chaku_kaisu_5", IsPrimaryKey = false, Comment = "新潟障着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1259, Length = 3, Name = "nigata_shogai_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "新潟障着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1262, Length = 3, Name = "tokyo_shogai_chaku_kaisu_1", IsPrimaryKey = false, Comment = "東京障着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1265, Length = 3, Name = "tokyo_shogai_chaku_kaisu_2", IsPrimaryKey = false, Comment = "東京障着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1268, Length = 3, Name = "tokyo_shogai_chaku_kaisu_3", IsPrimaryKey = false, Comment = "東京障着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1271, Length = 3, Name = "tokyo_shogai_chaku_kaisu_4", IsPrimaryKey = false, Comment = "東京障着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1274, Length = 3, Name = "tokyo_shogai_chaku_kaisu_5", IsPrimaryKey = false, Comment = "東京障着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1277, Length = 3, Name = "tokyo_shogai_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "東京障着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1280, Length = 3, Name = "nakayama_shogai_chaku_kaisu_1", IsPrimaryKey = false, Comment = "中山障着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1283, Length = 3, Name = "nakayama_shogai_chaku_kaisu_2", IsPrimaryKey = false, Comment = "中山障着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1286, Length = 3, Name = "nakayama_shogai_chaku_kaisu_3", IsPrimaryKey = false, Comment = "中山障着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1289, Length = 3, Name = "nakayama_shogai_chaku_kaisu_4", IsPrimaryKey = false, Comment = "中山障着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1292, Length = 3, Name = "nakayama_shogai_chaku_kaisu_5", IsPrimaryKey = false, Comment = "中山障着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1295, Length = 3, Name = "nakayama_shogai_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "中山障着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1298, Length = 3, Name = "chukyo_shogai_chaku_kaisu_1", IsPrimaryKey = false, Comment = "中京障着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1301, Length = 3, Name = "chukyo_shogai_chaku_kaisu_2", IsPrimaryKey = false, Comment = "中京障着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1304, Length = 3, Name = "chukyo_shogai_chaku_kaisu_3", IsPrimaryKey = false, Comment = "中京障着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1307, Length = 3, Name = "chukyo_shogai_chaku_kaisu_4", IsPrimaryKey = false, Comment = "中京障着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1310, Length = 3, Name = "chukyo_shogai_chaku_kaisu_5", IsPrimaryKey = false, Comment = "中京障着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1313, Length = 3, Name = "chukyo_shogai_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "中京障着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1316, Length = 3, Name = "kyoto_shogai_chaku_kaisu_1", IsPrimaryKey = false, Comment = "京都障着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1319, Length = 3, Name = "kyoto_shogai_chaku_kaisu_2", IsPrimaryKey = false, Comment = "京都障着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1322, Length = 3, Name = "kyoto_shogai_chaku_kaisu_3", IsPrimaryKey = false, Comment = "京都障着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1325, Length = 3, Name = "kyoto_shogai_chaku_kaisu_4", IsPrimaryKey = false, Comment = "京都障着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1328, Length = 3, Name = "kyoto_shogai_chaku_kaisu_5", IsPrimaryKey = false, Comment = "京都障着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1331, Length = 3, Name = "kyoto_shogai_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "京都障着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1334, Length = 3, Name = "hanshin_shogai_chaku_kaisu_1", IsPrimaryKey = false, Comment = "阪神障着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1337, Length = 3, Name = "hanshin_shogai_chaku_kaisu_2", IsPrimaryKey = false, Comment = "阪神障着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1340, Length = 3, Name = "hanshin_shogai_chaku_kaisu_3", IsPrimaryKey = false, Comment = "阪神障着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1343, Length = 3, Name = "hanshin_shogai_chaku_kaisu_4", IsPrimaryKey = false, Comment = "阪神障着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1346, Length = 3, Name = "hanshin_shogai_chaku_kaisu_5", IsPrimaryKey = false, Comment = "阪神障着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1349, Length = 3, Name = "hanshin_shogai_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "阪神障着回数着外", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1352, Length = 3, Name = "kokura_shogai_chaku_kaisu_1", IsPrimaryKey = false, Comment = "小倉障着回数1着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1355, Length = 3, Name = "kokura_shogai_chaku_kaisu_2", IsPrimaryKey = false, Comment = "小倉障着回数2着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1358, Length = 3, Name = "kokura_shogai_chaku_kaisu_3", IsPrimaryKey = false, Comment = "小倉障着回数3着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1361, Length = 3, Name = "kokura_shogai_chaku_kaisu_4", IsPrimaryKey = false, Comment = "小倉障着回数4着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1364, Length = 3, Name = "kokura_shogai_chaku_kaisu_5", IsPrimaryKey = false, Comment = "小倉障着回数5着", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1367, Length = 3, Name = "kokura_shogai_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "小倉障着回数着外", DataType = "CHAR" },
                        
                        // 脚質傾向 (4回繰り返し: 逃げ、先行、差し、追込)
                        new NormalFieldDefinition { Position = 1370, Length = 3, Name = "kyakushitsu_keiko_nige", IsPrimaryKey = false, Comment = "脚質傾向逃げ", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1373, Length = 3, Name = "kyakushitsu_keiko_senko", IsPrimaryKey = false, Comment = "脚質傾向先行", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1376, Length = 3, Name = "kyakushitsu_keiko_sashi", IsPrimaryKey = false, Comment = "脚質傾向差し", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1379, Length = 3, Name = "kyakushitsu_keiko_oikomi", IsPrimaryKey = false, Comment = "脚質傾向追込", DataType = "CHAR" },
                        
                        // 登録レース数
                        new NormalFieldDefinition { Position = 1382, Length = 3, Name = "toroku_race_su", IsPrimaryKey = false, Comment = "登録レース数", DataType = "CHAR" },
                        
                        // 騎手情報
                        new NormalFieldDefinition { Position = 1385, Length = 5, Name = "kishu_code", IsPrimaryKey = false, Comment = "騎手コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 1390, Length = 34, Name = "kishu_name", IsPrimaryKey = false, Comment = "騎手名", DataType = "CHAR" },

                        // 騎手本年・累計成績情報 (2回繰り返し: 本年、累計)
                        new RepeatFieldDefinition
                        {
                            Position = 1424,
                            Length = 1220,
                            RepeatCount = 2,
                            Table = new TableDefinition
                            {
                                Name = "kishu_seiseki_joho",
                                Comment = "騎手成績情報",
                                Fields = new List<FieldDefinition>
                                {
                                    new NormalFieldDefinition { Position = 1, Length = 4, Name = "settei_year", IsPrimaryKey = false, Comment = "設定年", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 5, Length = 10, Name = "heichi_hon_shokin_goukei", IsPrimaryKey = false, Comment = "平地本賞金合計", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 15, Length = 10, Name = "shogai_hon_shokin_goukei", IsPrimaryKey = false, Comment = "障害本賞金合計", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 25, Length = 10, Name = "heichi_fuka_shokin_goukei", IsPrimaryKey = false, Comment = "平地付加賞金合計", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 35, Length = 10, Name = "shogai_fuka_shokin_goukei", IsPrimaryKey = false, Comment = "障害付加賞金合計", DataType = "CHAR" },
                                    // 芝着回数 (6回繰り返し: 1-5着、着外)
                                    new NormalFieldDefinition { Position = 45, Length = 5, Name = "shiba_chaku_kaisu_1", IsPrimaryKey = false, Comment = "芝着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 50, Length = 5, Name = "shiba_chaku_kaisu_2", IsPrimaryKey = false, Comment = "芝着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 55, Length = 5, Name = "shiba_chaku_kaisu_3", IsPrimaryKey = false, Comment = "芝着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 60, Length = 5, Name = "shiba_chaku_kaisu_4", IsPrimaryKey = false, Comment = "芝着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 65, Length = 5, Name = "shiba_chaku_kaisu_5", IsPrimaryKey = false, Comment = "芝着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 70, Length = 5, Name = "shiba_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "芝着回数着外", DataType = "CHAR" },
                                    // ダート着回数 (6回繰り返し: 1-5着、着外)
                                    new NormalFieldDefinition { Position = 75, Length = 5, Name = "dirt_chaku_kaisu_1", IsPrimaryKey = false, Comment = "ダート着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 80, Length = 5, Name = "dirt_chaku_kaisu_2", IsPrimaryKey = false, Comment = "ダート着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 85, Length = 5, Name = "dirt_chaku_kaisu_3", IsPrimaryKey = false, Comment = "ダート着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 90, Length = 5, Name = "dirt_chaku_kaisu_4", IsPrimaryKey = false, Comment = "ダート着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 95, Length = 5, Name = "dirt_chaku_kaisu_5", IsPrimaryKey = false, Comment = "ダート着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 100, Length = 5, Name = "dirt_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "ダート着回数着外", DataType = "CHAR" },
                                    // 障害着回数 (6回繰り返し: 1-4着、着外) - 注意：障害は1-4着のみ
                                    new NormalFieldDefinition { Position = 105, Length = 4, Name = "shogai_chaku_kaisu_1", IsPrimaryKey = false, Comment = "障害着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 109, Length = 4, Name = "shogai_chaku_kaisu_2", IsPrimaryKey = false, Comment = "障害着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 113, Length = 4, Name = "shogai_chaku_kaisu_3", IsPrimaryKey = false, Comment = "障害着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 117, Length = 4, Name = "shogai_chaku_kaisu_4", IsPrimaryKey = false, Comment = "障害着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 121, Length = 4, Name = "shogai_chaku_kaisu_5", IsPrimaryKey = false, Comment = "障害着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 125, Length = 4, Name = "shogai_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "障害着回数着外", DataType = "CHAR" },
                                    // 芝距離別着回数
                                    new NormalFieldDefinition { Position = 129, Length = 4, Name = "shiba_1200_ika_chaku_kaisu_1", IsPrimaryKey = false, Comment = "芝1200以下着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 133, Length = 4, Name = "shiba_1200_ika_chaku_kaisu_2", IsPrimaryKey = false, Comment = "芝1200以下着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 137, Length = 4, Name = "shiba_1200_ika_chaku_kaisu_3", IsPrimaryKey = false, Comment = "芝1200以下着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 141, Length = 4, Name = "shiba_1200_ika_chaku_kaisu_4", IsPrimaryKey = false, Comment = "芝1200以下着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 145, Length = 4, Name = "shiba_1200_ika_chaku_kaisu_5", IsPrimaryKey = false, Comment = "芝1200以下着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 149, Length = 4, Name = "shiba_1200_ika_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "芝1200以下着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 153, Length = 4, Name = "shiba_1201_1400_chaku_kaisu_1", IsPrimaryKey = false, Comment = "芝1201-1400着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 157, Length = 4, Name = "shiba_1201_1400_chaku_kaisu_2", IsPrimaryKey = false, Comment = "芝1201-1400着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 161, Length = 4, Name = "shiba_1201_1400_chaku_kaisu_3", IsPrimaryKey = false, Comment = "芝1201-1400着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 165, Length = 4, Name = "shiba_1201_1400_chaku_kaisu_4", IsPrimaryKey = false, Comment = "芝1201-1400着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 169, Length = 4, Name = "shiba_1201_1400_chaku_kaisu_5", IsPrimaryKey = false, Comment = "芝1201-1400着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 173, Length = 4, Name = "shiba_1201_1400_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "芝1201-1400着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 177, Length = 4, Name = "shiba_1401_1600_chaku_kaisu_1", IsPrimaryKey = false, Comment = "芝1401-1600着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 181, Length = 4, Name = "shiba_1401_1600_chaku_kaisu_2", IsPrimaryKey = false, Comment = "芝1401-1600着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 185, Length = 4, Name = "shiba_1401_1600_chaku_kaisu_3", IsPrimaryKey = false, Comment = "芝1401-1600着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 189, Length = 4, Name = "shiba_1401_1600_chaku_kaisu_4", IsPrimaryKey = false, Comment = "芝1401-1600着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 193, Length = 4, Name = "shiba_1401_1600_chaku_kaisu_5", IsPrimaryKey = false, Comment = "芝1401-1600着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 197, Length = 4, Name = "shiba_1401_1600_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "芝1401-1600着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 201, Length = 4, Name = "shiba_1601_1800_chaku_kaisu_1", IsPrimaryKey = false, Comment = "芝1601-1800着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 205, Length = 4, Name = "shiba_1601_1800_chaku_kaisu_2", IsPrimaryKey = false, Comment = "芝1601-1800着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 209, Length = 4, Name = "shiba_1601_1800_chaku_kaisu_3", IsPrimaryKey = false, Comment = "芝1601-1800着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 213, Length = 4, Name = "shiba_1601_1800_chaku_kaisu_4", IsPrimaryKey = false, Comment = "芝1601-1800着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 217, Length = 4, Name = "shiba_1601_1800_chaku_kaisu_5", IsPrimaryKey = false, Comment = "芝1601-1800着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 221, Length = 4, Name = "shiba_1601_1800_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "芝1601-1800着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 225, Length = 4, Name = "shiba_1801_2000_chaku_kaisu_1", IsPrimaryKey = false, Comment = "芝1801-2000着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 229, Length = 4, Name = "shiba_1801_2000_chaku_kaisu_2", IsPrimaryKey = false, Comment = "芝1801-2000着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 233, Length = 4, Name = "shiba_1801_2000_chaku_kaisu_3", IsPrimaryKey = false, Comment = "芝1801-2000着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 237, Length = 4, Name = "shiba_1801_2000_chaku_kaisu_4", IsPrimaryKey = false, Comment = "芝1801-2000着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 241, Length = 4, Name = "shiba_1801_2000_chaku_kaisu_5", IsPrimaryKey = false, Comment = "芝1801-2000着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 245, Length = 4, Name = "shiba_1801_2000_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "芝1801-2000着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 249, Length = 4, Name = "shiba_2001_2200_chaku_kaisu_1", IsPrimaryKey = false, Comment = "芝2001-2200着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 253, Length = 4, Name = "shiba_2001_2200_chaku_kaisu_2", IsPrimaryKey = false, Comment = "芝2001-2200着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 257, Length = 4, Name = "shiba_2001_2200_chaku_kaisu_3", IsPrimaryKey = false, Comment = "芝2001-2200着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 261, Length = 4, Name = "shiba_2001_2200_chaku_kaisu_4", IsPrimaryKey = false, Comment = "芝2001-2200着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 265, Length = 4, Name = "shiba_2001_2200_chaku_kaisu_5", IsPrimaryKey = false, Comment = "芝2001-2200着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 269, Length = 4, Name = "shiba_2001_2200_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "芝2001-2200着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 273, Length = 4, Name = "shiba_2201_2400_chaku_kaisu_1", IsPrimaryKey = false, Comment = "芝2201-2400着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 277, Length = 4, Name = "shiba_2201_2400_chaku_kaisu_2", IsPrimaryKey = false, Comment = "芝2201-2400着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 281, Length = 4, Name = "shiba_2201_2400_chaku_kaisu_3", IsPrimaryKey = false, Comment = "芝2201-2400着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 285, Length = 4, Name = "shiba_2201_2400_chaku_kaisu_4", IsPrimaryKey = false, Comment = "芝2201-2400着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 289, Length = 4, Name = "shiba_2201_2400_chaku_kaisu_5", IsPrimaryKey = false, Comment = "芝2201-2400着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 293, Length = 4, Name = "shiba_2201_2400_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "芝2201-2400着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 297, Length = 4, Name = "shiba_2401_2800_chaku_kaisu_1", IsPrimaryKey = false, Comment = "芝2401-2800着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 301, Length = 4, Name = "shiba_2401_2800_chaku_kaisu_2", IsPrimaryKey = false, Comment = "芝2401-2800着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 305, Length = 4, Name = "shiba_2401_2800_chaku_kaisu_3", IsPrimaryKey = false, Comment = "芝2401-2800着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 309, Length = 4, Name = "shiba_2401_2800_chaku_kaisu_4", IsPrimaryKey = false, Comment = "芝2401-2800着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 313, Length = 4, Name = "shiba_2401_2800_chaku_kaisu_5", IsPrimaryKey = false, Comment = "芝2401-2800着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 317, Length = 4, Name = "shiba_2401_2800_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "芝2401-2800着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 321, Length = 4, Name = "shiba_2801_ijo_chaku_kaisu_1", IsPrimaryKey = false, Comment = "芝2801以上着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 325, Length = 4, Name = "shiba_2801_ijo_chaku_kaisu_2", IsPrimaryKey = false, Comment = "芝2801以上着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 329, Length = 4, Name = "shiba_2801_ijo_chaku_kaisu_3", IsPrimaryKey = false, Comment = "芝2801以上着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 333, Length = 4, Name = "shiba_2801_ijo_chaku_kaisu_4", IsPrimaryKey = false, Comment = "芝2801以上着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 337, Length = 4, Name = "shiba_2801_ijo_chaku_kaisu_5", IsPrimaryKey = false, Comment = "芝2801以上着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 341, Length = 4, Name = "shiba_2801_ijo_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "芝2801以上着回数着外", DataType = "CHAR" },
                                    // ダート距離別着回数
                                    new NormalFieldDefinition { Position = 345, Length = 4, Name = "dirt_1200_ika_chaku_kaisu_1", IsPrimaryKey = false, Comment = "ダート1200以下着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 349, Length = 4, Name = "dirt_1200_ika_chaku_kaisu_2", IsPrimaryKey = false, Comment = "ダート1200以下着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 353, Length = 4, Name = "dirt_1200_ika_chaku_kaisu_3", IsPrimaryKey = false, Comment = "ダート1200以下着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 357, Length = 4, Name = "dirt_1200_ika_chaku_kaisu_4", IsPrimaryKey = false, Comment = "ダート1200以下着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 361, Length = 4, Name = "dirt_1200_ika_chaku_kaisu_5", IsPrimaryKey = false, Comment = "ダート1200以下着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 365, Length = 4, Name = "dirt_1200_ika_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "ダート1200以下着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 369, Length = 4, Name = "dirt_1201_1400_chaku_kaisu_1", IsPrimaryKey = false, Comment = "ダート1201-1400着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 373, Length = 4, Name = "dirt_1201_1400_chaku_kaisu_2", IsPrimaryKey = false, Comment = "ダート1201-1400着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 377, Length = 4, Name = "dirt_1201_1400_chaku_kaisu_3", IsPrimaryKey = false, Comment = "ダート1201-1400着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 381, Length = 4, Name = "dirt_1201_1400_chaku_kaisu_4", IsPrimaryKey = false, Comment = "ダート1201-1400着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 385, Length = 4, Name = "dirt_1201_1400_chaku_kaisu_5", IsPrimaryKey = false, Comment = "ダート1201-1400着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 389, Length = 4, Name = "dirt_1201_1400_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "ダート1201-1400着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 393, Length = 4, Name = "dirt_1401_1600_chaku_kaisu_1", IsPrimaryKey = false, Comment = "ダート1401-1600着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 397, Length = 4, Name = "dirt_1401_1600_chaku_kaisu_2", IsPrimaryKey = false, Comment = "ダート1401-1600着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 401, Length = 4, Name = "dirt_1401_1600_chaku_kaisu_3", IsPrimaryKey = false, Comment = "ダート1401-1600着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 405, Length = 4, Name = "dirt_1401_1600_chaku_kaisu_4", IsPrimaryKey = false, Comment = "ダート1401-1600着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 409, Length = 4, Name = "dirt_1401_1600_chaku_kaisu_5", IsPrimaryKey = false, Comment = "ダート1401-1600着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 413, Length = 4, Name = "dirt_1401_1600_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "ダート1401-1600着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 417, Length = 4, Name = "dirt_1601_1800_chaku_kaisu_1", IsPrimaryKey = false, Comment = "ダート1601-1800着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 421, Length = 4, Name = "dirt_1601_1800_chaku_kaisu_2", IsPrimaryKey = false, Comment = "ダート1601-1800着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 425, Length = 4, Name = "dirt_1601_1800_chaku_kaisu_3", IsPrimaryKey = false, Comment = "ダート1601-1800着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 429, Length = 4, Name = "dirt_1601_1800_chaku_kaisu_4", IsPrimaryKey = false, Comment = "ダート1601-1800着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 433, Length = 4, Name = "dirt_1601_1800_chaku_kaisu_5", IsPrimaryKey = false, Comment = "ダート1601-1800着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 437, Length = 4, Name = "dirt_1601_1800_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "ダート1601-1800着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 441, Length = 4, Name = "dirt_1801_2000_chaku_kaisu_1", IsPrimaryKey = false, Comment = "ダート1801-2000着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 445, Length = 4, Name = "dirt_1801_2000_chaku_kaisu_2", IsPrimaryKey = false, Comment = "ダート1801-2000着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 449, Length = 4, Name = "dirt_1801_2000_chaku_kaisu_3", IsPrimaryKey = false, Comment = "ダート1801-2000着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 453, Length = 4, Name = "dirt_1801_2000_chaku_kaisu_4", IsPrimaryKey = false, Comment = "ダート1801-2000着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 457, Length = 4, Name = "dirt_1801_2000_chaku_kaisu_5", IsPrimaryKey = false, Comment = "ダート1801-2000着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 461, Length = 4, Name = "dirt_1801_2000_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "ダート1801-2000着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 465, Length = 4, Name = "dirt_2001_2200_chaku_kaisu_1", IsPrimaryKey = false, Comment = "ダート2001-2200着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 469, Length = 4, Name = "dirt_2001_2200_chaku_kaisu_2", IsPrimaryKey = false, Comment = "ダート2001-2200着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 473, Length = 4, Name = "dirt_2001_2200_chaku_kaisu_3", IsPrimaryKey = false, Comment = "ダート2001-2200着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 477, Length = 4, Name = "dirt_2001_2200_chaku_kaisu_4", IsPrimaryKey = false, Comment = "ダート2001-2200着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 481, Length = 4, Name = "dirt_2001_2200_chaku_kaisu_5", IsPrimaryKey = false, Comment = "ダート2001-2200着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 485, Length = 4, Name = "dirt_2001_2200_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "ダート2001-2200着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 489, Length = 4, Name = "dirt_2201_2400_chaku_kaisu_1", IsPrimaryKey = false, Comment = "ダート2201-2400着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 493, Length = 4, Name = "dirt_2201_2400_chaku_kaisu_2", IsPrimaryKey = false, Comment = "ダート2201-2400着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 497, Length = 4, Name = "dirt_2201_2400_chaku_kaisu_3", IsPrimaryKey = false, Comment = "ダート2201-2400着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 501, Length = 4, Name = "dirt_2201_2400_chaku_kaisu_4", IsPrimaryKey = false, Comment = "ダート2201-2400着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 505, Length = 4, Name = "dirt_2201_2400_chaku_kaisu_5", IsPrimaryKey = false, Comment = "ダート2201-2400着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 509, Length = 4, Name = "dirt_2201_2400_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "ダート2201-2400着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 513, Length = 4, Name = "dirt_2401_2800_chaku_kaisu_1", IsPrimaryKey = false, Comment = "ダート2401-2800着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 517, Length = 4, Name = "dirt_2401_2800_chaku_kaisu_2", IsPrimaryKey = false, Comment = "ダート2401-2800着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 521, Length = 4, Name = "dirt_2401_2800_chaku_kaisu_3", IsPrimaryKey = false, Comment = "ダート2401-2800着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 525, Length = 4, Name = "dirt_2401_2800_chaku_kaisu_4", IsPrimaryKey = false, Comment = "ダート2401-2800着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 529, Length = 4, Name = "dirt_2401_2800_chaku_kaisu_5", IsPrimaryKey = false, Comment = "ダート2401-2800着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 533, Length = 4, Name = "dirt_2401_2800_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "ダート2401-2800着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 537, Length = 4, Name = "dirt_2801_ijo_chaku_kaisu_1", IsPrimaryKey = false, Comment = "ダート2801以上着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 541, Length = 4, Name = "dirt_2801_ijo_chaku_kaisu_2", IsPrimaryKey = false, Comment = "ダート2801以上着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 545, Length = 4, Name = "dirt_2801_ijo_chaku_kaisu_3", IsPrimaryKey = false, Comment = "ダート2801以上着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 549, Length = 4, Name = "dirt_2801_ijo_chaku_kaisu_4", IsPrimaryKey = false, Comment = "ダート2801以上着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 553, Length = 4, Name = "dirt_2801_ijo_chaku_kaisu_5", IsPrimaryKey = false, Comment = "ダート2801以上着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 557, Length = 4, Name = "dirt_2801_ijo_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "ダート2801以上着回数着外", DataType = "CHAR" },
                                    // 競馬場別芝着回数
                                    new NormalFieldDefinition { Position = 561, Length = 4, Name = "sapporo_shiba_chaku_kaisu_1", IsPrimaryKey = false, Comment = "札幌芝着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 565, Length = 4, Name = "sapporo_shiba_chaku_kaisu_2", IsPrimaryKey = false, Comment = "札幌芝着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 569, Length = 4, Name = "sapporo_shiba_chaku_kaisu_3", IsPrimaryKey = false, Comment = "札幌芝着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 573, Length = 4, Name = "sapporo_shiba_chaku_kaisu_4", IsPrimaryKey = false, Comment = "札幌芝着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 577, Length = 4, Name = "sapporo_shiba_chaku_kaisu_5", IsPrimaryKey = false, Comment = "札幌芝着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 581, Length = 4, Name = "sapporo_shiba_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "札幌芝着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 585, Length = 4, Name = "hakodate_shiba_chaku_kaisu_1", IsPrimaryKey = false, Comment = "函館芝着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 589, Length = 4, Name = "hakodate_shiba_chaku_kaisu_2", IsPrimaryKey = false, Comment = "函館芝着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 593, Length = 4, Name = "hakodate_shiba_chaku_kaisu_3", IsPrimaryKey = false, Comment = "函館芝着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 597, Length = 4, Name = "hakodate_shiba_chaku_kaisu_4", IsPrimaryKey = false, Comment = "函館芝着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 601, Length = 4, Name = "hakodate_shiba_chaku_kaisu_5", IsPrimaryKey = false, Comment = "函館芝着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 605, Length = 4, Name = "hakodate_shiba_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "函館芝着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 609, Length = 4, Name = "fukushima_shiba_chaku_kaisu_1", IsPrimaryKey = false, Comment = "福島芝着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 613, Length = 4, Name = "fukushima_shiba_chaku_kaisu_2", IsPrimaryKey = false, Comment = "福島芝着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 617, Length = 4, Name = "fukushima_shiba_chaku_kaisu_3", IsPrimaryKey = false, Comment = "福島芝着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 621, Length = 4, Name = "fukushima_shiba_chaku_kaisu_4", IsPrimaryKey = false, Comment = "福島芝着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 625, Length = 4, Name = "fukushima_shiba_chaku_kaisu_5", IsPrimaryKey = false, Comment = "福島芝着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 629, Length = 4, Name = "fukushima_shiba_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "福島芝着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 633, Length = 4, Name = "nigata_shiba_chaku_kaisu_1", IsPrimaryKey = false, Comment = "新潟芝着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 637, Length = 4, Name = "nigata_shiba_chaku_kaisu_2", IsPrimaryKey = false, Comment = "新潟芝着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 641, Length = 4, Name = "nigata_shiba_chaku_kaisu_3", IsPrimaryKey = false, Comment = "新潟芝着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 645, Length = 4, Name = "nigata_shiba_chaku_kaisu_4", IsPrimaryKey = false, Comment = "新潟芝着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 649, Length = 4, Name = "nigata_shiba_chaku_kaisu_5", IsPrimaryKey = false, Comment = "新潟芝着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 653, Length = 4, Name = "nigata_shiba_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "新潟芝着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 657, Length = 4, Name = "tokyo_shiba_chaku_kaisu_1", IsPrimaryKey = false, Comment = "東京芝着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 661, Length = 4, Name = "tokyo_shiba_chaku_kaisu_2", IsPrimaryKey = false, Comment = "東京芝着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 665, Length = 4, Name = "tokyo_shiba_chaku_kaisu_3", IsPrimaryKey = false, Comment = "東京芝着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 669, Length = 4, Name = "tokyo_shiba_chaku_kaisu_4", IsPrimaryKey = false, Comment = "東京芝着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 673, Length = 4, Name = "tokyo_shiba_chaku_kaisu_5", IsPrimaryKey = false, Comment = "東京芝着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 677, Length = 4, Name = "tokyo_shiba_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "東京芝着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 681, Length = 4, Name = "nakayama_shiba_chaku_kaisu_1", IsPrimaryKey = false, Comment = "中山芝着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 685, Length = 4, Name = "nakayama_shiba_chaku_kaisu_2", IsPrimaryKey = false, Comment = "中山芝着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 689, Length = 4, Name = "nakayama_shiba_chaku_kaisu_3", IsPrimaryKey = false, Comment = "中山芝着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 693, Length = 4, Name = "nakayama_shiba_chaku_kaisu_4", IsPrimaryKey = false, Comment = "中山芝着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 697, Length = 4, Name = "nakayama_shiba_chaku_kaisu_5", IsPrimaryKey = false, Comment = "中山芝着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 701, Length = 4, Name = "nakayama_shiba_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "中山芝着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 705, Length = 4, Name = "chukyo_shiba_chaku_kaisu_1", IsPrimaryKey = false, Comment = "中京芝着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 709, Length = 4, Name = "chukyo_shiba_chaku_kaisu_2", IsPrimaryKey = false, Comment = "中京芝着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 713, Length = 4, Name = "chukyo_shiba_chaku_kaisu_3", IsPrimaryKey = false, Comment = "中京芝着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 717, Length = 4, Name = "chukyo_shiba_chaku_kaisu_4", IsPrimaryKey = false, Comment = "中京芝着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 721, Length = 4, Name = "chukyo_shiba_chaku_kaisu_5", IsPrimaryKey = false, Comment = "中京芝着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 725, Length = 4, Name = "chukyo_shiba_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "中京芝着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 729, Length = 4, Name = "kyoto_shiba_chaku_kaisu_1", IsPrimaryKey = false, Comment = "京都芝着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 733, Length = 4, Name = "kyoto_shiba_chaku_kaisu_2", IsPrimaryKey = false, Comment = "京都芝着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 737, Length = 4, Name = "kyoto_shiba_chaku_kaisu_3", IsPrimaryKey = false, Comment = "京都芝着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 741, Length = 4, Name = "kyoto_shiba_chaku_kaisu_4", IsPrimaryKey = false, Comment = "京都芝着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 745, Length = 4, Name = "kyoto_shiba_chaku_kaisu_5", IsPrimaryKey = false, Comment = "京都芝着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 749, Length = 4, Name = "kyoto_shiba_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "京都芝着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 753, Length = 4, Name = "hanshin_shiba_chaku_kaisu_1", IsPrimaryKey = false, Comment = "阪神芝着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 757, Length = 4, Name = "hanshin_shiba_chaku_kaisu_2", IsPrimaryKey = false, Comment = "阪神芝着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 761, Length = 4, Name = "hanshin_shiba_chaku_kaisu_3", IsPrimaryKey = false, Comment = "阪神芝着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 765, Length = 4, Name = "hanshin_shiba_chaku_kaisu_4", IsPrimaryKey = false, Comment = "阪神芝着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 769, Length = 4, Name = "hanshin_shiba_chaku_kaisu_5", IsPrimaryKey = false, Comment = "阪神芝着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 773, Length = 4, Name = "hanshin_shiba_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "阪神芝着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 777, Length = 4, Name = "kokura_shiba_chaku_kaisu_1", IsPrimaryKey = false, Comment = "小倉芝着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 781, Length = 4, Name = "kokura_shiba_chaku_kaisu_2", IsPrimaryKey = false, Comment = "小倉芝着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 785, Length = 4, Name = "kokura_shiba_chaku_kaisu_3", IsPrimaryKey = false, Comment = "小倉芝着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 789, Length = 4, Name = "kokura_shiba_chaku_kaisu_4", IsPrimaryKey = false, Comment = "小倉芝着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 793, Length = 4, Name = "kokura_shiba_chaku_kaisu_5", IsPrimaryKey = false, Comment = "小倉芝着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 797, Length = 4, Name = "kokura_shiba_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "小倉芝着回数着外", DataType = "CHAR" },
                                    // 競馬場別ダート着回数
                                    new NormalFieldDefinition { Position = 801, Length = 4, Name = "sapporo_dirt_chaku_kaisu_1", IsPrimaryKey = false, Comment = "札幌ダート着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 805, Length = 4, Name = "sapporo_dirt_chaku_kaisu_2", IsPrimaryKey = false, Comment = "札幌ダート着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 809, Length = 4, Name = "sapporo_dirt_chaku_kaisu_3", IsPrimaryKey = false, Comment = "札幌ダート着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 813, Length = 4, Name = "sapporo_dirt_chaku_kaisu_4", IsPrimaryKey = false, Comment = "札幌ダート着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 817, Length = 4, Name = "sapporo_dirt_chaku_kaisu_5", IsPrimaryKey = false, Comment = "札幌ダート着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 821, Length = 4, Name = "sapporo_dirt_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "札幌ダート着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 825, Length = 4, Name = "hakodate_dirt_chaku_kaisu_1", IsPrimaryKey = false, Comment = "函館ダート着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 829, Length = 4, Name = "hakodate_dirt_chaku_kaisu_2", IsPrimaryKey = false, Comment = "函館ダート着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 833, Length = 4, Name = "hakodate_dirt_chaku_kaisu_3", IsPrimaryKey = false, Comment = "函館ダート着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 837, Length = 4, Name = "hakodate_dirt_chaku_kaisu_4", IsPrimaryKey = false, Comment = "函館ダート着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 841, Length = 4, Name = "hakodate_dirt_chaku_kaisu_5", IsPrimaryKey = false, Comment = "函館ダート着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 845, Length = 4, Name = "hakodate_dirt_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "函館ダート着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 849, Length = 4, Name = "fukushima_dirt_chaku_kaisu_1", IsPrimaryKey = false, Comment = "福島ダート着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 853, Length = 4, Name = "fukushima_dirt_chaku_kaisu_2", IsPrimaryKey = false, Comment = "福島ダート着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 857, Length = 4, Name = "fukushima_dirt_chaku_kaisu_3", IsPrimaryKey = false, Comment = "福島ダート着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 861, Length = 4, Name = "fukushima_dirt_chaku_kaisu_4", IsPrimaryKey = false, Comment = "福島ダート着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 865, Length = 4, Name = "fukushima_dirt_chaku_kaisu_5", IsPrimaryKey = false, Comment = "福島ダート着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 869, Length = 4, Name = "fukushima_dirt_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "福島ダート着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 873, Length = 4, Name = "nigata_dirt_chaku_kaisu_1", IsPrimaryKey = false, Comment = "新潟ダート着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 877, Length = 4, Name = "nigata_dirt_chaku_kaisu_2", IsPrimaryKey = false, Comment = "新潟ダート着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 881, Length = 4, Name = "nigata_dirt_chaku_kaisu_3", IsPrimaryKey = false, Comment = "新潟ダート着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 885, Length = 4, Name = "nigata_dirt_chaku_kaisu_4", IsPrimaryKey = false, Comment = "新潟ダート着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 889, Length = 4, Name = "nigata_dirt_chaku_kaisu_5", IsPrimaryKey = false, Comment = "新潟ダート着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 893, Length = 4, Name = "nigata_dirt_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "新潟ダート着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 897, Length = 4, Name = "tokyo_dirt_chaku_kaisu_1", IsPrimaryKey = false, Comment = "東京ダート着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 901, Length = 4, Name = "tokyo_dirt_chaku_kaisu_2", IsPrimaryKey = false, Comment = "東京ダート着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 905, Length = 4, Name = "tokyo_dirt_chaku_kaisu_3", IsPrimaryKey = false, Comment = "東京ダート着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 909, Length = 4, Name = "tokyo_dirt_chaku_kaisu_4", IsPrimaryKey = false, Comment = "東京ダート着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 913, Length = 4, Name = "tokyo_dirt_chaku_kaisu_5", IsPrimaryKey = false, Comment = "東京ダート着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 917, Length = 4, Name = "tokyo_dirt_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "東京ダート着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 921, Length = 4, Name = "nakayama_dirt_chaku_kaisu_1", IsPrimaryKey = false, Comment = "中山ダート着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 925, Length = 4, Name = "nakayama_dirt_chaku_kaisu_2", IsPrimaryKey = false, Comment = "中山ダート着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 929, Length = 4, Name = "nakayama_dirt_chaku_kaisu_3", IsPrimaryKey = false, Comment = "中山ダート着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 933, Length = 4, Name = "nakayama_dirt_chaku_kaisu_4", IsPrimaryKey = false, Comment = "中山ダート着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 937, Length = 4, Name = "nakayama_dirt_chaku_kaisu_5", IsPrimaryKey = false, Comment = "中山ダート着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 941, Length = 4, Name = "nakayama_dirt_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "中山ダート着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 945, Length = 4, Name = "chukyo_dirt_chaku_kaisu_1", IsPrimaryKey = false, Comment = "中京ダート着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 949, Length = 4, Name = "chukyo_dirt_chaku_kaisu_2", IsPrimaryKey = false, Comment = "中京ダート着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 953, Length = 4, Name = "chukyo_dirt_chaku_kaisu_3", IsPrimaryKey = false, Comment = "中京ダート着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 957, Length = 4, Name = "chukyo_dirt_chaku_kaisu_4", IsPrimaryKey = false, Comment = "中京ダート着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 961, Length = 4, Name = "chukyo_dirt_chaku_kaisu_5", IsPrimaryKey = false, Comment = "中京ダート着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 965, Length = 4, Name = "chukyo_dirt_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "中京ダート着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 969, Length = 4, Name = "kyoto_dirt_chaku_kaisu_1", IsPrimaryKey = false, Comment = "京都ダート着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 973, Length = 4, Name = "kyoto_dirt_chaku_kaisu_2", IsPrimaryKey = false, Comment = "京都ダート着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 977, Length = 4, Name = "kyoto_dirt_chaku_kaisu_3", IsPrimaryKey = false, Comment = "京都ダート着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 981, Length = 4, Name = "kyoto_dirt_chaku_kaisu_4", IsPrimaryKey = false, Comment = "京都ダート着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 985, Length = 4, Name = "kyoto_dirt_chaku_kaisu_5", IsPrimaryKey = false, Comment = "京都ダート着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 989, Length = 4, Name = "kyoto_dirt_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "京都ダート着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 993, Length = 4, Name = "hanshin_dirt_chaku_kaisu_1", IsPrimaryKey = false, Comment = "阪神ダート着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 997, Length = 4, Name = "hanshin_dirt_chaku_kaisu_2", IsPrimaryKey = false, Comment = "阪神ダート着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1001, Length = 4, Name = "hanshin_dirt_chaku_kaisu_3", IsPrimaryKey = false, Comment = "阪神ダート着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1005, Length = 4, Name = "hanshin_dirt_chaku_kaisu_4", IsPrimaryKey = false, Comment = "阪神ダート着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1009, Length = 4, Name = "hanshin_dirt_chaku_kaisu_5", IsPrimaryKey = false, Comment = "阪神ダート着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1013, Length = 4, Name = "hanshin_dirt_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "阪神ダート着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1017, Length = 4, Name = "kokura_dirt_chaku_kaisu_1", IsPrimaryKey = false, Comment = "小倉ダート着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1021, Length = 4, Name = "kokura_dirt_chaku_kaisu_2", IsPrimaryKey = false, Comment = "小倉ダート着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1025, Length = 4, Name = "kokura_dirt_chaku_kaisu_3", IsPrimaryKey = false, Comment = "小倉ダート着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1029, Length = 4, Name = "kokura_dirt_chaku_kaisu_4", IsPrimaryKey = false, Comment = "小倉ダート着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1033, Length = 4, Name = "kokura_dirt_chaku_kaisu_5", IsPrimaryKey = false, Comment = "小倉ダート着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1037, Length = 4, Name = "kokura_dirt_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "小倉ダート着回数着外", DataType = "CHAR" },
                                    // 競馬場別障害着回数
                                    new NormalFieldDefinition { Position = 1041, Length = 3, Name = "sapporo_shogai_chaku_kaisu_1", IsPrimaryKey = false, Comment = "札幌障害着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1044, Length = 3, Name = "sapporo_shogai_chaku_kaisu_2", IsPrimaryKey = false, Comment = "札幌障害着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1047, Length = 3, Name = "sapporo_shogai_chaku_kaisu_3", IsPrimaryKey = false, Comment = "札幌障害着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1050, Length = 3, Name = "sapporo_shogai_chaku_kaisu_4", IsPrimaryKey = false, Comment = "札幌障害着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1053, Length = 3, Name = "sapporo_shogai_chaku_kaisu_5", IsPrimaryKey = false, Comment = "札幌障害着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1056, Length = 3, Name = "sapporo_shogai_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "札幌障害着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1059, Length = 3, Name = "hakodate_shogai_chaku_kaisu_1", IsPrimaryKey = false, Comment = "函館障害着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1062, Length = 3, Name = "hakodate_shogai_chaku_kaisu_2", IsPrimaryKey = false, Comment = "函館障害着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1065, Length = 3, Name = "hakodate_shogai_chaku_kaisu_3", IsPrimaryKey = false, Comment = "函館障害着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1068, Length = 3, Name = "hakodate_shogai_chaku_kaisu_4", IsPrimaryKey = false, Comment = "函館障害着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1071, Length = 3, Name = "hakodate_shogai_chaku_kaisu_5", IsPrimaryKey = false, Comment = "函館障害着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1074, Length = 3, Name = "hakodate_shogai_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "函館障害着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1077, Length = 3, Name = "fukushima_shogai_chaku_kaisu_1", IsPrimaryKey = false, Comment = "福島障害着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1080, Length = 3, Name = "fukushima_shogai_chaku_kaisu_2", IsPrimaryKey = false, Comment = "福島障害着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1083, Length = 3, Name = "fukushima_shogai_chaku_kaisu_3", IsPrimaryKey = false, Comment = "福島障害着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1086, Length = 3, Name = "fukushima_shogai_chaku_kaisu_4", IsPrimaryKey = false, Comment = "福島障害着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1089, Length = 3, Name = "fukushima_shogai_chaku_kaisu_5", IsPrimaryKey = false, Comment = "福島障害着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1092, Length = 3, Name = "fukushima_shogai_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "福島障害着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1095, Length = 3, Name = "nigata_shogai_chaku_kaisu_1", IsPrimaryKey = false, Comment = "新潟障害着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1098, Length = 3, Name = "nigata_shogai_chaku_kaisu_2", IsPrimaryKey = false, Comment = "新潟障害着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1101, Length = 3, Name = "nigata_shogai_chaku_kaisu_3", IsPrimaryKey = false, Comment = "新潟障害着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1104, Length = 3, Name = "nigata_shogai_chaku_kaisu_4", IsPrimaryKey = false, Comment = "新潟障害着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1107, Length = 3, Name = "nigata_shogai_chaku_kaisu_5", IsPrimaryKey = false, Comment = "新潟障害着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1110, Length = 3, Name = "nigata_shogai_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "新潟障害着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1113, Length = 3, Name = "tokyo_shogai_chaku_kaisu_1", IsPrimaryKey = false, Comment = "東京障害着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1116, Length = 3, Name = "tokyo_shogai_chaku_kaisu_2", IsPrimaryKey = false, Comment = "東京障害着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1119, Length = 3, Name = "tokyo_shogai_chaku_kaisu_3", IsPrimaryKey = false, Comment = "東京障害着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1122, Length = 3, Name = "tokyo_shogai_chaku_kaisu_4", IsPrimaryKey = false, Comment = "東京障害着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1125, Length = 3, Name = "tokyo_shogai_chaku_kaisu_5", IsPrimaryKey = false, Comment = "東京障害着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1128, Length = 3, Name = "tokyo_shogai_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "東京障害着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1131, Length = 3, Name = "nakayama_shogai_chaku_kaisu_1", IsPrimaryKey = false, Comment = "中山障害着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1134, Length = 3, Name = "nakayama_shogai_chaku_kaisu_2", IsPrimaryKey = false, Comment = "中山障害着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1137, Length = 3, Name = "nakayama_shogai_chaku_kaisu_3", IsPrimaryKey = false, Comment = "中山障害着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1140, Length = 3, Name = "nakayama_shogai_chaku_kaisu_4", IsPrimaryKey = false, Comment = "中山障害着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1143, Length = 3, Name = "nakayama_shogai_chaku_kaisu_5", IsPrimaryKey = false, Comment = "中山障害着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1146, Length = 3, Name = "nakayama_shogai_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "中山障害着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1149, Length = 3, Name = "chukyo_shogai_chaku_kaisu_1", IsPrimaryKey = false, Comment = "中京障害着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1152, Length = 3, Name = "chukyo_shogai_chaku_kaisu_2", IsPrimaryKey = false, Comment = "中京障害着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1155, Length = 3, Name = "chukyo_shogai_chaku_kaisu_3", IsPrimaryKey = false, Comment = "中京障害着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1158, Length = 3, Name = "chukyo_shogai_chaku_kaisu_4", IsPrimaryKey = false, Comment = "中京障害着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1161, Length = 3, Name = "chukyo_shogai_chaku_kaisu_5", IsPrimaryKey = false, Comment = "中京障害着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1164, Length = 3, Name = "chukyo_shogai_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "中京障害着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1167, Length = 3, Name = "kyoto_shogai_chaku_kaisu_1", IsPrimaryKey = false, Comment = "京都障害着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1170, Length = 3, Name = "kyoto_shogai_chaku_kaisu_2", IsPrimaryKey = false, Comment = "京都障害着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1173, Length = 3, Name = "kyoto_shogai_chaku_kaisu_3", IsPrimaryKey = false, Comment = "京都障害着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1176, Length = 3, Name = "kyoto_shogai_chaku_kaisu_4", IsPrimaryKey = false, Comment = "京都障害着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1179, Length = 3, Name = "kyoto_shogai_chaku_kaisu_5", IsPrimaryKey = false, Comment = "京都障害着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1182, Length = 3, Name = "kyoto_shogai_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "京都障害着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1185, Length = 3, Name = "hanshin_shogai_chaku_kaisu_1", IsPrimaryKey = false, Comment = "阪神障害着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1188, Length = 3, Name = "hanshin_shogai_chaku_kaisu_2", IsPrimaryKey = false, Comment = "阪神障害着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1191, Length = 3, Name = "hanshin_shogai_chaku_kaisu_3", IsPrimaryKey = false, Comment = "阪神障害着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1194, Length = 3, Name = "hanshin_shogai_chaku_kaisu_4", IsPrimaryKey = false, Comment = "阪神障害着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1197, Length = 3, Name = "hanshin_shogai_chaku_kaisu_5", IsPrimaryKey = false, Comment = "阪神障害着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1200, Length = 3, Name = "hanshin_shogai_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "阪神障害着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1203, Length = 3, Name = "kokura_shogai_chaku_kaisu_1", IsPrimaryKey = false, Comment = "小倉障害着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1206, Length = 3, Name = "kokura_shogai_chaku_kaisu_2", IsPrimaryKey = false, Comment = "小倉障害着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1209, Length = 3, Name = "kokura_shogai_chaku_kaisu_3", IsPrimaryKey = false, Comment = "小倉障害着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1212, Length = 3, Name = "kokura_shogai_chaku_kaisu_4", IsPrimaryKey = false, Comment = "小倉障害着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1215, Length = 3, Name = "kokura_shogai_chaku_kaisu_5", IsPrimaryKey = false, Comment = "小倉障害着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1218, Length = 3, Name = "kokura_shogai_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "小倉障害着回数着外", DataType = "CHAR" }
                                }
                            }
                        },

                        // 調教師情報
                        new NormalFieldDefinition { Position = 3864, Length = 5, Name = "chokyoshi_code", IsPrimaryKey = false, Comment = "調教師コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 3869, Length = 34, Name = "chokyoshi_name", IsPrimaryKey = false, Comment = "調教師名", DataType = "CHAR" },
                        
                        // 調教師本年・累計成績情報 (2回繰り返し: 本年、累計)
                        new RepeatFieldDefinition
                        {
                            Position = 3903,
                            Length = 1220,
                            RepeatCount = 2,
                            Table = new TableDefinition
                            {
                                Name = "chokyoshi_seiseki_joho",
                                Comment = "調教師成績情報",
                                Fields = new List<FieldDefinition>
                                {
                                    new NormalFieldDefinition { Position = 1, Length = 4, Name = "settei_year", IsPrimaryKey = false, Comment = "設定年", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 5, Length = 10, Name = "heichi_hon_shokin_goukei", IsPrimaryKey = false, Comment = "平地本賞金合計", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 15, Length = 10, Name = "shogai_hon_shokin_goukei", IsPrimaryKey = false, Comment = "障害本賞金合計", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 25, Length = 10, Name = "heichi_fuka_shokin_goukei", IsPrimaryKey = false, Comment = "平地付加賞金合計", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 35, Length = 10, Name = "shogai_fuka_shokin_goukei", IsPrimaryKey = false, Comment = "障害付加賞金合計", DataType = "CHAR" },
                                    // 芝着回数 (6回繰り返し: 1-5着、着外)
                                    new NormalFieldDefinition { Position = 45, Length = 5, Name = "shiba_chaku_kaisu_1", IsPrimaryKey = false, Comment = "芝着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 50, Length = 5, Name = "shiba_chaku_kaisu_2", IsPrimaryKey = false, Comment = "芝着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 55, Length = 5, Name = "shiba_chaku_kaisu_3", IsPrimaryKey = false, Comment = "芝着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 60, Length = 5, Name = "shiba_chaku_kaisu_4", IsPrimaryKey = false, Comment = "芝着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 65, Length = 5, Name = "shiba_chaku_kaisu_5", IsPrimaryKey = false, Comment = "芝着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 70, Length = 5, Name = "shiba_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "芝着回数着外", DataType = "CHAR" },
                                    // ダート着回数 (6回繰り返し: 1-5着、着外)
                                    new NormalFieldDefinition { Position = 75, Length = 5, Name = "dirt_chaku_kaisu_1", IsPrimaryKey = false, Comment = "ダート着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 80, Length = 5, Name = "dirt_chaku_kaisu_2", IsPrimaryKey = false, Comment = "ダート着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 85, Length = 5, Name = "dirt_chaku_kaisu_3", IsPrimaryKey = false, Comment = "ダート着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 90, Length = 5, Name = "dirt_chaku_kaisu_4", IsPrimaryKey = false, Comment = "ダート着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 95, Length = 5, Name = "dirt_chaku_kaisu_5", IsPrimaryKey = false, Comment = "ダート着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 100, Length = 5, Name = "dirt_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "ダート着回数着外", DataType = "CHAR" },
                                    // 障害着回数 (6回繰り返し: 1-4着、着外) - 注意：障害は1-4着のみ
                                    new NormalFieldDefinition { Position = 105, Length = 4, Name = "shogai_chaku_kaisu_1", IsPrimaryKey = false, Comment = "障害着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 109, Length = 4, Name = "shogai_chaku_kaisu_2", IsPrimaryKey = false, Comment = "障害着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 113, Length = 4, Name = "shogai_chaku_kaisu_3", IsPrimaryKey = false, Comment = "障害着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 117, Length = 4, Name = "shogai_chaku_kaisu_4", IsPrimaryKey = false, Comment = "障害着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 121, Length = 4, Name = "shogai_chaku_kaisu_5", IsPrimaryKey = false, Comment = "障害着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 125, Length = 4, Name = "shogai_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "障害着回数着外", DataType = "CHAR" },
                                    // 芝距離別着回数
                                    new NormalFieldDefinition { Position = 129, Length = 4, Name = "shiba_1200_ika_chaku_kaisu_1", IsPrimaryKey = false, Comment = "芝1200以下着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 133, Length = 4, Name = "shiba_1200_ika_chaku_kaisu_2", IsPrimaryKey = false, Comment = "芝1200以下着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 137, Length = 4, Name = "shiba_1200_ika_chaku_kaisu_3", IsPrimaryKey = false, Comment = "芝1200以下着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 141, Length = 4, Name = "shiba_1200_ika_chaku_kaisu_4", IsPrimaryKey = false, Comment = "芝1200以下着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 145, Length = 4, Name = "shiba_1200_ika_chaku_kaisu_5", IsPrimaryKey = false, Comment = "芝1200以下着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 149, Length = 4, Name = "shiba_1200_ika_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "芝1200以下着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 153, Length = 4, Name = "shiba_1201_1400_chaku_kaisu_1", IsPrimaryKey = false, Comment = "芝1201-1400着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 157, Length = 4, Name = "shiba_1201_1400_chaku_kaisu_2", IsPrimaryKey = false, Comment = "芝1201-1400着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 161, Length = 4, Name = "shiba_1201_1400_chaku_kaisu_3", IsPrimaryKey = false, Comment = "芝1201-1400着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 165, Length = 4, Name = "shiba_1201_1400_chaku_kaisu_4", IsPrimaryKey = false, Comment = "芝1201-1400着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 169, Length = 4, Name = "shiba_1201_1400_chaku_kaisu_5", IsPrimaryKey = false, Comment = "芝1201-1400着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 173, Length = 4, Name = "shiba_1201_1400_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "芝1201-1400着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 177, Length = 4, Name = "shiba_1401_1600_chaku_kaisu_1", IsPrimaryKey = false, Comment = "芝1401-1600着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 181, Length = 4, Name = "shiba_1401_1600_chaku_kaisu_2", IsPrimaryKey = false, Comment = "芝1401-1600着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 185, Length = 4, Name = "shiba_1401_1600_chaku_kaisu_3", IsPrimaryKey = false, Comment = "芝1401-1600着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 189, Length = 4, Name = "shiba_1401_1600_chaku_kaisu_4", IsPrimaryKey = false, Comment = "芝1401-1600着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 193, Length = 4, Name = "shiba_1401_1600_chaku_kaisu_5", IsPrimaryKey = false, Comment = "芝1401-1600着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 197, Length = 4, Name = "shiba_1401_1600_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "芝1401-1600着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 201, Length = 4, Name = "shiba_1601_1800_chaku_kaisu_1", IsPrimaryKey = false, Comment = "芝1601-1800着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 205, Length = 4, Name = "shiba_1601_1800_chaku_kaisu_2", IsPrimaryKey = false, Comment = "芝1601-1800着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 209, Length = 4, Name = "shiba_1601_1800_chaku_kaisu_3", IsPrimaryKey = false, Comment = "芝1601-1800着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 213, Length = 4, Name = "shiba_1601_1800_chaku_kaisu_4", IsPrimaryKey = false, Comment = "芝1601-1800着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 217, Length = 4, Name = "shiba_1601_1800_chaku_kaisu_5", IsPrimaryKey = false, Comment = "芝1601-1800着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 221, Length = 4, Name = "shiba_1601_1800_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "芝1601-1800着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 225, Length = 4, Name = "shiba_1801_2000_chaku_kaisu_1", IsPrimaryKey = false, Comment = "芝1801-2000着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 229, Length = 4, Name = "shiba_1801_2000_chaku_kaisu_2", IsPrimaryKey = false, Comment = "芝1801-2000着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 233, Length = 4, Name = "shiba_1801_2000_chaku_kaisu_3", IsPrimaryKey = false, Comment = "芝1801-2000着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 237, Length = 4, Name = "shiba_1801_2000_chaku_kaisu_4", IsPrimaryKey = false, Comment = "芝1801-2000着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 241, Length = 4, Name = "shiba_1801_2000_chaku_kaisu_5", IsPrimaryKey = false, Comment = "芝1801-2000着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 245, Length = 4, Name = "shiba_1801_2000_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "芝1801-2000着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 249, Length = 4, Name = "shiba_2001_2200_chaku_kaisu_1", IsPrimaryKey = false, Comment = "芝2001-2200着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 253, Length = 4, Name = "shiba_2001_2200_chaku_kaisu_2", IsPrimaryKey = false, Comment = "芝2001-2200着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 257, Length = 4, Name = "shiba_2001_2200_chaku_kaisu_3", IsPrimaryKey = false, Comment = "芝2001-2200着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 261, Length = 4, Name = "shiba_2001_2200_chaku_kaisu_4", IsPrimaryKey = false, Comment = "芝2001-2200着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 265, Length = 4, Name = "shiba_2001_2200_chaku_kaisu_5", IsPrimaryKey = false, Comment = "芝2001-2200着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 269, Length = 4, Name = "shiba_2001_2200_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "芝2001-2200着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 273, Length = 4, Name = "shiba_2201_2400_chaku_kaisu_1", IsPrimaryKey = false, Comment = "芝2201-2400着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 277, Length = 4, Name = "shiba_2201_2400_chaku_kaisu_2", IsPrimaryKey = false, Comment = "芝2201-2400着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 281, Length = 4, Name = "shiba_2201_2400_chaku_kaisu_3", IsPrimaryKey = false, Comment = "芝2201-2400着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 285, Length = 4, Name = "shiba_2201_2400_chaku_kaisu_4", IsPrimaryKey = false, Comment = "芝2201-2400着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 289, Length = 4, Name = "shiba_2201_2400_chaku_kaisu_5", IsPrimaryKey = false, Comment = "芝2201-2400着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 293, Length = 4, Name = "shiba_2201_2400_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "芝2201-2400着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 297, Length = 4, Name = "shiba_2401_2800_chaku_kaisu_1", IsPrimaryKey = false, Comment = "芝2401-2800着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 301, Length = 4, Name = "shiba_2401_2800_chaku_kaisu_2", IsPrimaryKey = false, Comment = "芝2401-2800着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 305, Length = 4, Name = "shiba_2401_2800_chaku_kaisu_3", IsPrimaryKey = false, Comment = "芝2401-2800着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 309, Length = 4, Name = "shiba_2401_2800_chaku_kaisu_4", IsPrimaryKey = false, Comment = "芝2401-2800着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 313, Length = 4, Name = "shiba_2401_2800_chaku_kaisu_5", IsPrimaryKey = false, Comment = "芝2401-2800着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 317, Length = 4, Name = "shiba_2401_2800_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "芝2401-2800着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 321, Length = 4, Name = "shiba_2801_ijo_chaku_kaisu_1", IsPrimaryKey = false, Comment = "芝2801以上着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 325, Length = 4, Name = "shiba_2801_ijo_chaku_kaisu_2", IsPrimaryKey = false, Comment = "芝2801以上着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 329, Length = 4, Name = "shiba_2801_ijo_chaku_kaisu_3", IsPrimaryKey = false, Comment = "芝2801以上着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 333, Length = 4, Name = "shiba_2801_ijo_chaku_kaisu_4", IsPrimaryKey = false, Comment = "芝2801以上着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 337, Length = 4, Name = "shiba_2801_ijo_chaku_kaisu_5", IsPrimaryKey = false, Comment = "芝2801以上着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 341, Length = 4, Name = "shiba_2801_ijo_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "芝2801以上着回数着外", DataType = "CHAR" },
                                    // ダート距離別着回数
                                    new NormalFieldDefinition { Position = 345, Length = 4, Name = "dirt_1200_ika_chaku_kaisu_1", IsPrimaryKey = false, Comment = "ダート1200以下着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 349, Length = 4, Name = "dirt_1200_ika_chaku_kaisu_2", IsPrimaryKey = false, Comment = "ダート1200以下着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 353, Length = 4, Name = "dirt_1200_ika_chaku_kaisu_3", IsPrimaryKey = false, Comment = "ダート1200以下着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 357, Length = 4, Name = "dirt_1200_ika_chaku_kaisu_4", IsPrimaryKey = false, Comment = "ダート1200以下着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 361, Length = 4, Name = "dirt_1200_ika_chaku_kaisu_5", IsPrimaryKey = false, Comment = "ダート1200以下着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 365, Length = 4, Name = "dirt_1200_ika_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "ダート1200以下着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 369, Length = 4, Name = "dirt_1201_1400_chaku_kaisu_1", IsPrimaryKey = false, Comment = "ダート1201-1400着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 373, Length = 4, Name = "dirt_1201_1400_chaku_kaisu_2", IsPrimaryKey = false, Comment = "ダート1201-1400着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 377, Length = 4, Name = "dirt_1201_1400_chaku_kaisu_3", IsPrimaryKey = false, Comment = "ダート1201-1400着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 381, Length = 4, Name = "dirt_1201_1400_chaku_kaisu_4", IsPrimaryKey = false, Comment = "ダート1201-1400着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 385, Length = 4, Name = "dirt_1201_1400_chaku_kaisu_5", IsPrimaryKey = false, Comment = "ダート1201-1400着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 389, Length = 4, Name = "dirt_1201_1400_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "ダート1201-1400着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 393, Length = 4, Name = "dirt_1401_1600_chaku_kaisu_1", IsPrimaryKey = false, Comment = "ダート1401-1600着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 397, Length = 4, Name = "dirt_1401_1600_chaku_kaisu_2", IsPrimaryKey = false, Comment = "ダート1401-1600着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 401, Length = 4, Name = "dirt_1401_1600_chaku_kaisu_3", IsPrimaryKey = false, Comment = "ダート1401-1600着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 405, Length = 4, Name = "dirt_1401_1600_chaku_kaisu_4", IsPrimaryKey = false, Comment = "ダート1401-1600着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 409, Length = 4, Name = "dirt_1401_1600_chaku_kaisu_5", IsPrimaryKey = false, Comment = "ダート1401-1600着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 413, Length = 4, Name = "dirt_1401_1600_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "ダート1401-1600着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 417, Length = 4, Name = "dirt_1601_1800_chaku_kaisu_1", IsPrimaryKey = false, Comment = "ダート1601-1800着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 421, Length = 4, Name = "dirt_1601_1800_chaku_kaisu_2", IsPrimaryKey = false, Comment = "ダート1601-1800着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 425, Length = 4, Name = "dirt_1601_1800_chaku_kaisu_3", IsPrimaryKey = false, Comment = "ダート1601-1800着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 429, Length = 4, Name = "dirt_1601_1800_chaku_kaisu_4", IsPrimaryKey = false, Comment = "ダート1601-1800着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 433, Length = 4, Name = "dirt_1601_1800_chaku_kaisu_5", IsPrimaryKey = false, Comment = "ダート1601-1800着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 437, Length = 4, Name = "dirt_1601_1800_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "ダート1601-1800着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 441, Length = 4, Name = "dirt_1801_2000_chaku_kaisu_1", IsPrimaryKey = false, Comment = "ダート1801-2000着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 445, Length = 4, Name = "dirt_1801_2000_chaku_kaisu_2", IsPrimaryKey = false, Comment = "ダート1801-2000着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 449, Length = 4, Name = "dirt_1801_2000_chaku_kaisu_3", IsPrimaryKey = false, Comment = "ダート1801-2000着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 453, Length = 4, Name = "dirt_1801_2000_chaku_kaisu_4", IsPrimaryKey = false, Comment = "ダート1801-2000着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 457, Length = 4, Name = "dirt_1801_2000_chaku_kaisu_5", IsPrimaryKey = false, Comment = "ダート1801-2000着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 461, Length = 4, Name = "dirt_1801_2000_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "ダート1801-2000着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 465, Length = 4, Name = "dirt_2001_2200_chaku_kaisu_1", IsPrimaryKey = false, Comment = "ダート2001-2200着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 469, Length = 4, Name = "dirt_2001_2200_chaku_kaisu_2", IsPrimaryKey = false, Comment = "ダート2001-2200着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 473, Length = 4, Name = "dirt_2001_2200_chaku_kaisu_3", IsPrimaryKey = false, Comment = "ダート2001-2200着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 477, Length = 4, Name = "dirt_2001_2200_chaku_kaisu_4", IsPrimaryKey = false, Comment = "ダート2001-2200着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 481, Length = 4, Name = "dirt_2001_2200_chaku_kaisu_5", IsPrimaryKey = false, Comment = "ダート2001-2200着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 485, Length = 4, Name = "dirt_2001_2200_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "ダート2001-2200着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 489, Length = 4, Name = "dirt_2201_2400_chaku_kaisu_1", IsPrimaryKey = false, Comment = "ダート2201-2400着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 493, Length = 4, Name = "dirt_2201_2400_chaku_kaisu_2", IsPrimaryKey = false, Comment = "ダート2201-2400着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 497, Length = 4, Name = "dirt_2201_2400_chaku_kaisu_3", IsPrimaryKey = false, Comment = "ダート2201-2400着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 501, Length = 4, Name = "dirt_2201_2400_chaku_kaisu_4", IsPrimaryKey = false, Comment = "ダート2201-2400着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 505, Length = 4, Name = "dirt_2201_2400_chaku_kaisu_5", IsPrimaryKey = false, Comment = "ダート2201-2400着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 509, Length = 4, Name = "dirt_2201_2400_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "ダート2201-2400着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 513, Length = 4, Name = "dirt_2401_2800_chaku_kaisu_1", IsPrimaryKey = false, Comment = "ダート2401-2800着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 517, Length = 4, Name = "dirt_2401_2800_chaku_kaisu_2", IsPrimaryKey = false, Comment = "ダート2401-2800着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 521, Length = 4, Name = "dirt_2401_2800_chaku_kaisu_3", IsPrimaryKey = false, Comment = "ダート2401-2800着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 525, Length = 4, Name = "dirt_2401_2800_chaku_kaisu_4", IsPrimaryKey = false, Comment = "ダート2401-2800着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 529, Length = 4, Name = "dirt_2401_2800_chaku_kaisu_5", IsPrimaryKey = false, Comment = "ダート2401-2800着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 533, Length = 4, Name = "dirt_2401_2800_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "ダート2401-2800着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 537, Length = 4, Name = "dirt_2801_ijo_chaku_kaisu_1", IsPrimaryKey = false, Comment = "ダート2801以上着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 541, Length = 4, Name = "dirt_2801_ijo_chaku_kaisu_2", IsPrimaryKey = false, Comment = "ダート2801以上着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 545, Length = 4, Name = "dirt_2801_ijo_chaku_kaisu_3", IsPrimaryKey = false, Comment = "ダート2801以上着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 549, Length = 4, Name = "dirt_2801_ijo_chaku_kaisu_4", IsPrimaryKey = false, Comment = "ダート2801以上着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 553, Length = 4, Name = "dirt_2801_ijo_chaku_kaisu_5", IsPrimaryKey = false, Comment = "ダート2801以上着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 557, Length = 4, Name = "dirt_2801_ijo_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "ダート2801以上着回数着外", DataType = "CHAR" },
                                    // 競馬場別芝着回数
                                    new NormalFieldDefinition { Position = 561, Length = 4, Name = "sapporo_shiba_chaku_kaisu_1", IsPrimaryKey = false, Comment = "札幌芝着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 565, Length = 4, Name = "sapporo_shiba_chaku_kaisu_2", IsPrimaryKey = false, Comment = "札幌芝着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 569, Length = 4, Name = "sapporo_shiba_chaku_kaisu_3", IsPrimaryKey = false, Comment = "札幌芝着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 573, Length = 4, Name = "sapporo_shiba_chaku_kaisu_4", IsPrimaryKey = false, Comment = "札幌芝着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 577, Length = 4, Name = "sapporo_shiba_chaku_kaisu_5", IsPrimaryKey = false, Comment = "札幌芝着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 581, Length = 4, Name = "sapporo_shiba_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "札幌芝着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 585, Length = 4, Name = "hakodate_shiba_chaku_kaisu_1", IsPrimaryKey = false, Comment = "函館芝着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 589, Length = 4, Name = "hakodate_shiba_chaku_kaisu_2", IsPrimaryKey = false, Comment = "函館芝着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 593, Length = 4, Name = "hakodate_shiba_chaku_kaisu_3", IsPrimaryKey = false, Comment = "函館芝着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 597, Length = 4, Name = "hakodate_shiba_chaku_kaisu_4", IsPrimaryKey = false, Comment = "函館芝着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 601, Length = 4, Name = "hakodate_shiba_chaku_kaisu_5", IsPrimaryKey = false, Comment = "函館芝着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 605, Length = 4, Name = "hakodate_shiba_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "函館芝着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 609, Length = 4, Name = "fukushima_shiba_chaku_kaisu_1", IsPrimaryKey = false, Comment = "福島芝着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 613, Length = 4, Name = "fukushima_shiba_chaku_kaisu_2", IsPrimaryKey = false, Comment = "福島芝着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 617, Length = 4, Name = "fukushima_shiba_chaku_kaisu_3", IsPrimaryKey = false, Comment = "福島芝着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 621, Length = 4, Name = "fukushima_shiba_chaku_kaisu_4", IsPrimaryKey = false, Comment = "福島芝着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 625, Length = 4, Name = "fukushima_shiba_chaku_kaisu_5", IsPrimaryKey = false, Comment = "福島芝着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 629, Length = 4, Name = "fukushima_shiba_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "福島芝着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 633, Length = 4, Name = "nigata_shiba_chaku_kaisu_1", IsPrimaryKey = false, Comment = "新潟芝着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 637, Length = 4, Name = "nigata_shiba_chaku_kaisu_2", IsPrimaryKey = false, Comment = "新潟芝着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 641, Length = 4, Name = "nigata_shiba_chaku_kaisu_3", IsPrimaryKey = false, Comment = "新潟芝着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 645, Length = 4, Name = "nigata_shiba_chaku_kaisu_4", IsPrimaryKey = false, Comment = "新潟芝着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 649, Length = 4, Name = "nigata_shiba_chaku_kaisu_5", IsPrimaryKey = false, Comment = "新潟芝着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 653, Length = 4, Name = "nigata_shiba_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "新潟芝着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 657, Length = 4, Name = "tokyo_shiba_chaku_kaisu_1", IsPrimaryKey = false, Comment = "東京芝着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 661, Length = 4, Name = "tokyo_shiba_chaku_kaisu_2", IsPrimaryKey = false, Comment = "東京芝着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 665, Length = 4, Name = "tokyo_shiba_chaku_kaisu_3", IsPrimaryKey = false, Comment = "東京芝着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 669, Length = 4, Name = "tokyo_shiba_chaku_kaisu_4", IsPrimaryKey = false, Comment = "東京芝着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 673, Length = 4, Name = "tokyo_shiba_chaku_kaisu_5", IsPrimaryKey = false, Comment = "東京芝着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 677, Length = 4, Name = "tokyo_shiba_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "東京芝着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 681, Length = 4, Name = "nakayama_shiba_chaku_kaisu_1", IsPrimaryKey = false, Comment = "中山芝着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 685, Length = 4, Name = "nakayama_shiba_chaku_kaisu_2", IsPrimaryKey = false, Comment = "中山芝着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 689, Length = 4, Name = "nakayama_shiba_chaku_kaisu_3", IsPrimaryKey = false, Comment = "中山芝着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 693, Length = 4, Name = "nakayama_shiba_chaku_kaisu_4", IsPrimaryKey = false, Comment = "中山芝着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 697, Length = 4, Name = "nakayama_shiba_chaku_kaisu_5", IsPrimaryKey = false, Comment = "中山芝着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 701, Length = 4, Name = "nakayama_shiba_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "中山芝着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 705, Length = 4, Name = "chukyo_shiba_chaku_kaisu_1", IsPrimaryKey = false, Comment = "中京芝着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 709, Length = 4, Name = "chukyo_shiba_chaku_kaisu_2", IsPrimaryKey = false, Comment = "中京芝着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 713, Length = 4, Name = "chukyo_shiba_chaku_kaisu_3", IsPrimaryKey = false, Comment = "中京芝着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 717, Length = 4, Name = "chukyo_shiba_chaku_kaisu_4", IsPrimaryKey = false, Comment = "中京芝着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 721, Length = 4, Name = "chukyo_shiba_chaku_kaisu_5", IsPrimaryKey = false, Comment = "中京芝着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 725, Length = 4, Name = "chukyo_shiba_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "中京芝着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 729, Length = 4, Name = "kyoto_shiba_chaku_kaisu_1", IsPrimaryKey = false, Comment = "京都芝着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 733, Length = 4, Name = "kyoto_shiba_chaku_kaisu_2", IsPrimaryKey = false, Comment = "京都芝着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 737, Length = 4, Name = "kyoto_shiba_chaku_kaisu_3", IsPrimaryKey = false, Comment = "京都芝着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 741, Length = 4, Name = "kyoto_shiba_chaku_kaisu_4", IsPrimaryKey = false, Comment = "京都芝着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 745, Length = 4, Name = "kyoto_shiba_chaku_kaisu_5", IsPrimaryKey = false, Comment = "京都芝着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 749, Length = 4, Name = "kyoto_shiba_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "京都芝着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 753, Length = 4, Name = "hanshin_shiba_chaku_kaisu_1", IsPrimaryKey = false, Comment = "阪神芝着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 757, Length = 4, Name = "hanshin_shiba_chaku_kaisu_2", IsPrimaryKey = false, Comment = "阪神芝着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 761, Length = 4, Name = "hanshin_shiba_chaku_kaisu_3", IsPrimaryKey = false, Comment = "阪神芝着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 765, Length = 4, Name = "hanshin_shiba_chaku_kaisu_4", IsPrimaryKey = false, Comment = "阪神芝着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 769, Length = 4, Name = "hanshin_shiba_chaku_kaisu_5", IsPrimaryKey = false, Comment = "阪神芝着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 773, Length = 4, Name = "hanshin_shiba_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "阪神芝着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 777, Length = 4, Name = "kokura_shiba_chaku_kaisu_1", IsPrimaryKey = false, Comment = "小倉芝着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 781, Length = 4, Name = "kokura_shiba_chaku_kaisu_2", IsPrimaryKey = false, Comment = "小倉芝着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 785, Length = 4, Name = "kokura_shiba_chaku_kaisu_3", IsPrimaryKey = false, Comment = "小倉芝着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 789, Length = 4, Name = "kokura_shiba_chaku_kaisu_4", IsPrimaryKey = false, Comment = "小倉芝着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 793, Length = 4, Name = "kokura_shiba_chaku_kaisu_5", IsPrimaryKey = false, Comment = "小倉芝着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 797, Length = 4, Name = "kokura_shiba_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "小倉芝着回数着外", DataType = "CHAR" },
                                    // 競馬場別ダート着回数
                                    new NormalFieldDefinition { Position = 801, Length = 4, Name = "sapporo_dirt_chaku_kaisu_1", IsPrimaryKey = false, Comment = "札幌ダート着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 805, Length = 4, Name = "sapporo_dirt_chaku_kaisu_2", IsPrimaryKey = false, Comment = "札幌ダート着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 809, Length = 4, Name = "sapporo_dirt_chaku_kaisu_3", IsPrimaryKey = false, Comment = "札幌ダート着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 813, Length = 4, Name = "sapporo_dirt_chaku_kaisu_4", IsPrimaryKey = false, Comment = "札幌ダート着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 817, Length = 4, Name = "sapporo_dirt_chaku_kaisu_5", IsPrimaryKey = false, Comment = "札幌ダート着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 821, Length = 4, Name = "sapporo_dirt_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "札幌ダート着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 825, Length = 4, Name = "hakodate_dirt_chaku_kaisu_1", IsPrimaryKey = false, Comment = "函館ダート着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 829, Length = 4, Name = "hakodate_dirt_chaku_kaisu_2", IsPrimaryKey = false, Comment = "函館ダート着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 833, Length = 4, Name = "hakodate_dirt_chaku_kaisu_3", IsPrimaryKey = false, Comment = "函館ダート着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 837, Length = 4, Name = "hakodate_dirt_chaku_kaisu_4", IsPrimaryKey = false, Comment = "函館ダート着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 841, Length = 4, Name = "hakodate_dirt_chaku_kaisu_5", IsPrimaryKey = false, Comment = "函館ダート着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 845, Length = 4, Name = "hakodate_dirt_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "函館ダート着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 849, Length = 4, Name = "fukushima_dirt_chaku_kaisu_1", IsPrimaryKey = false, Comment = "福島ダート着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 853, Length = 4, Name = "fukushima_dirt_chaku_kaisu_2", IsPrimaryKey = false, Comment = "福島ダート着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 857, Length = 4, Name = "fukushima_dirt_chaku_kaisu_3", IsPrimaryKey = false, Comment = "福島ダート着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 861, Length = 4, Name = "fukushima_dirt_chaku_kaisu_4", IsPrimaryKey = false, Comment = "福島ダート着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 865, Length = 4, Name = "fukushima_dirt_chaku_kaisu_5", IsPrimaryKey = false, Comment = "福島ダート着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 869, Length = 4, Name = "fukushima_dirt_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "福島ダート着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 873, Length = 4, Name = "nigata_dirt_chaku_kaisu_1", IsPrimaryKey = false, Comment = "新潟ダート着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 877, Length = 4, Name = "nigata_dirt_chaku_kaisu_2", IsPrimaryKey = false, Comment = "新潟ダート着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 881, Length = 4, Name = "nigata_dirt_chaku_kaisu_3", IsPrimaryKey = false, Comment = "新潟ダート着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 885, Length = 4, Name = "nigata_dirt_chaku_kaisu_4", IsPrimaryKey = false, Comment = "新潟ダート着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 889, Length = 4, Name = "nigata_dirt_chaku_kaisu_5", IsPrimaryKey = false, Comment = "新潟ダート着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 893, Length = 4, Name = "nigata_dirt_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "新潟ダート着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 897, Length = 4, Name = "tokyo_dirt_chaku_kaisu_1", IsPrimaryKey = false, Comment = "東京ダート着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 901, Length = 4, Name = "tokyo_dirt_chaku_kaisu_2", IsPrimaryKey = false, Comment = "東京ダート着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 905, Length = 4, Name = "tokyo_dirt_chaku_kaisu_3", IsPrimaryKey = false, Comment = "東京ダート着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 909, Length = 4, Name = "tokyo_dirt_chaku_kaisu_4", IsPrimaryKey = false, Comment = "東京ダート着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 913, Length = 4, Name = "tokyo_dirt_chaku_kaisu_5", IsPrimaryKey = false, Comment = "東京ダート着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 917, Length = 4, Name = "tokyo_dirt_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "東京ダート着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 921, Length = 4, Name = "nakayama_dirt_chaku_kaisu_1", IsPrimaryKey = false, Comment = "中山ダート着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 925, Length = 4, Name = "nakayama_dirt_chaku_kaisu_2", IsPrimaryKey = false, Comment = "中山ダート着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 929, Length = 4, Name = "nakayama_dirt_chaku_kaisu_3", IsPrimaryKey = false, Comment = "中山ダート着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 933, Length = 4, Name = "nakayama_dirt_chaku_kaisu_4", IsPrimaryKey = false, Comment = "中山ダート着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 937, Length = 4, Name = "nakayama_dirt_chaku_kaisu_5", IsPrimaryKey = false, Comment = "中山ダート着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 941, Length = 4, Name = "nakayama_dirt_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "中山ダート着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 945, Length = 4, Name = "chukyo_dirt_chaku_kaisu_1", IsPrimaryKey = false, Comment = "中京ダート着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 949, Length = 4, Name = "chukyo_dirt_chaku_kaisu_2", IsPrimaryKey = false, Comment = "中京ダート着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 953, Length = 4, Name = "chukyo_dirt_chaku_kaisu_3", IsPrimaryKey = false, Comment = "中京ダート着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 957, Length = 4, Name = "chukyo_dirt_chaku_kaisu_4", IsPrimaryKey = false, Comment = "中京ダート着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 961, Length = 4, Name = "chukyo_dirt_chaku_kaisu_5", IsPrimaryKey = false, Comment = "中京ダート着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 965, Length = 4, Name = "chukyo_dirt_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "中京ダート着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 969, Length = 4, Name = "kyoto_dirt_chaku_kaisu_1", IsPrimaryKey = false, Comment = "京都ダート着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 973, Length = 4, Name = "kyoto_dirt_chaku_kaisu_2", IsPrimaryKey = false, Comment = "京都ダート着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 977, Length = 4, Name = "kyoto_dirt_chaku_kaisu_3", IsPrimaryKey = false, Comment = "京都ダート着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 981, Length = 4, Name = "kyoto_dirt_chaku_kaisu_4", IsPrimaryKey = false, Comment = "京都ダート着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 985, Length = 4, Name = "kyoto_dirt_chaku_kaisu_5", IsPrimaryKey = false, Comment = "京都ダート着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 989, Length = 4, Name = "kyoto_dirt_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "京都ダート着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 993, Length = 4, Name = "hanshin_dirt_chaku_kaisu_1", IsPrimaryKey = false, Comment = "阪神ダート着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 997, Length = 4, Name = "hanshin_dirt_chaku_kaisu_2", IsPrimaryKey = false, Comment = "阪神ダート着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1001, Length = 4, Name = "hanshin_dirt_chaku_kaisu_3", IsPrimaryKey = false, Comment = "阪神ダート着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1005, Length = 4, Name = "hanshin_dirt_chaku_kaisu_4", IsPrimaryKey = false, Comment = "阪神ダート着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1009, Length = 4, Name = "hanshin_dirt_chaku_kaisu_5", IsPrimaryKey = false, Comment = "阪神ダート着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1013, Length = 4, Name = "hanshin_dirt_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "阪神ダート着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1017, Length = 4, Name = "kokura_dirt_chaku_kaisu_1", IsPrimaryKey = false, Comment = "小倉ダート着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1021, Length = 4, Name = "kokura_dirt_chaku_kaisu_2", IsPrimaryKey = false, Comment = "小倉ダート着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1025, Length = 4, Name = "kokura_dirt_chaku_kaisu_3", IsPrimaryKey = false, Comment = "小倉ダート着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1029, Length = 4, Name = "kokura_dirt_chaku_kaisu_4", IsPrimaryKey = false, Comment = "小倉ダート着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1033, Length = 4, Name = "kokura_dirt_chaku_kaisu_5", IsPrimaryKey = false, Comment = "小倉ダート着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1037, Length = 4, Name = "kokura_dirt_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "小倉ダート着回数着外", DataType = "CHAR" },
                                    // 競馬場別障害着回数
                                    new NormalFieldDefinition { Position = 1041, Length = 3, Name = "sapporo_shogai_chaku_kaisu_1", IsPrimaryKey = false, Comment = "札幌障害着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1044, Length = 3, Name = "sapporo_shogai_chaku_kaisu_2", IsPrimaryKey = false, Comment = "札幌障害着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1047, Length = 3, Name = "sapporo_shogai_chaku_kaisu_3", IsPrimaryKey = false, Comment = "札幌障害着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1050, Length = 3, Name = "sapporo_shogai_chaku_kaisu_4", IsPrimaryKey = false, Comment = "札幌障害着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1053, Length = 3, Name = "sapporo_shogai_chaku_kaisu_5", IsPrimaryKey = false, Comment = "札幌障害着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1056, Length = 3, Name = "sapporo_shogai_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "札幌障害着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1059, Length = 3, Name = "hakodate_shogai_chaku_kaisu_1", IsPrimaryKey = false, Comment = "函館障害着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1062, Length = 3, Name = "hakodate_shogai_chaku_kaisu_2", IsPrimaryKey = false, Comment = "函館障害着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1065, Length = 3, Name = "hakodate_shogai_chaku_kaisu_3", IsPrimaryKey = false, Comment = "函館障害着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1068, Length = 3, Name = "hakodate_shogai_chaku_kaisu_4", IsPrimaryKey = false, Comment = "函館障害着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1071, Length = 3, Name = "hakodate_shogai_chaku_kaisu_5", IsPrimaryKey = false, Comment = "函館障害着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1074, Length = 3, Name = "hakodate_shogai_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "函館障害着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1077, Length = 3, Name = "fukushima_shogai_chaku_kaisu_1", IsPrimaryKey = false, Comment = "福島障害着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1080, Length = 3, Name = "fukushima_shogai_chaku_kaisu_2", IsPrimaryKey = false, Comment = "福島障害着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1083, Length = 3, Name = "fukushima_shogai_chaku_kaisu_3", IsPrimaryKey = false, Comment = "福島障害着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1086, Length = 3, Name = "fukushima_shogai_chaku_kaisu_4", IsPrimaryKey = false, Comment = "福島障害着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1089, Length = 3, Name = "fukushima_shogai_chaku_kaisu_5", IsPrimaryKey = false, Comment = "福島障害着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1092, Length = 3, Name = "fukushima_shogai_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "福島障害着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1095, Length = 3, Name = "nigata_shogai_chaku_kaisu_1", IsPrimaryKey = false, Comment = "新潟障害着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1098, Length = 3, Name = "nigata_shogai_chaku_kaisu_2", IsPrimaryKey = false, Comment = "新潟障害着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1101, Length = 3, Name = "nigata_shogai_chaku_kaisu_3", IsPrimaryKey = false, Comment = "新潟障害着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1104, Length = 3, Name = "nigata_shogai_chaku_kaisu_4", IsPrimaryKey = false, Comment = "新潟障害着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1107, Length = 3, Name = "nigata_shogai_chaku_kaisu_5", IsPrimaryKey = false, Comment = "新潟障害着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1110, Length = 3, Name = "nigata_shogai_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "新潟障害着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1113, Length = 3, Name = "tokyo_shogai_chaku_kaisu_1", IsPrimaryKey = false, Comment = "東京障害着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1116, Length = 3, Name = "tokyo_shogai_chaku_kaisu_2", IsPrimaryKey = false, Comment = "東京障害着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1119, Length = 3, Name = "tokyo_shogai_chaku_kaisu_3", IsPrimaryKey = false, Comment = "東京障害着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1122, Length = 3, Name = "tokyo_shogai_chaku_kaisu_4", IsPrimaryKey = false, Comment = "東京障害着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1125, Length = 3, Name = "tokyo_shogai_chaku_kaisu_5", IsPrimaryKey = false, Comment = "東京障害着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1128, Length = 3, Name = "tokyo_shogai_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "東京障害着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1131, Length = 3, Name = "nakayama_shogai_chaku_kaisu_1", IsPrimaryKey = false, Comment = "中山障害着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1134, Length = 3, Name = "nakayama_shogai_chaku_kaisu_2", IsPrimaryKey = false, Comment = "中山障害着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1137, Length = 3, Name = "nakayama_shogai_chaku_kaisu_3", IsPrimaryKey = false, Comment = "中山障害着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1140, Length = 3, Name = "nakayama_shogai_chaku_kaisu_4", IsPrimaryKey = false, Comment = "中山障害着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1143, Length = 3, Name = "nakayama_shogai_chaku_kaisu_5", IsPrimaryKey = false, Comment = "中山障害着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1146, Length = 3, Name = "nakayama_shogai_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "中山障害着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1149, Length = 3, Name = "chukyo_shogai_chaku_kaisu_1", IsPrimaryKey = false, Comment = "中京障害着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1152, Length = 3, Name = "chukyo_shogai_chaku_kaisu_2", IsPrimaryKey = false, Comment = "中京障害着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1155, Length = 3, Name = "chukyo_shogai_chaku_kaisu_3", IsPrimaryKey = false, Comment = "中京障害着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1158, Length = 3, Name = "chukyo_shogai_chaku_kaisu_4", IsPrimaryKey = false, Comment = "中京障害着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1161, Length = 3, Name = "chukyo_shogai_chaku_kaisu_5", IsPrimaryKey = false, Comment = "中京障害着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1164, Length = 3, Name = "chukyo_shogai_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "中京障害着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1167, Length = 3, Name = "kyoto_shogai_chaku_kaisu_1", IsPrimaryKey = false, Comment = "京都障害着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1170, Length = 3, Name = "kyoto_shogai_chaku_kaisu_2", IsPrimaryKey = false, Comment = "京都障害着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1173, Length = 3, Name = "kyoto_shogai_chaku_kaisu_3", IsPrimaryKey = false, Comment = "京都障害着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1176, Length = 3, Name = "kyoto_shogai_chaku_kaisu_4", IsPrimaryKey = false, Comment = "京都障害着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1179, Length = 3, Name = "kyoto_shogai_chaku_kaisu_5", IsPrimaryKey = false, Comment = "京都障害着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1182, Length = 3, Name = "kyoto_shogai_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "京都障害着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1185, Length = 3, Name = "hanshin_shogai_chaku_kaisu_1", IsPrimaryKey = false, Comment = "阪神障害着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1188, Length = 3, Name = "hanshin_shogai_chaku_kaisu_2", IsPrimaryKey = false, Comment = "阪神障害着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1191, Length = 3, Name = "hanshin_shogai_chaku_kaisu_3", IsPrimaryKey = false, Comment = "阪神障害着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1194, Length = 3, Name = "hanshin_shogai_chaku_kaisu_4", IsPrimaryKey = false, Comment = "阪神障害着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1197, Length = 3, Name = "hanshin_shogai_chaku_kaisu_5", IsPrimaryKey = false, Comment = "阪神障害着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1200, Length = 3, Name = "hanshin_shogai_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "阪神障害着回数着外", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1203, Length = 3, Name = "kokura_shogai_chaku_kaisu_1", IsPrimaryKey = false, Comment = "小倉障害着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1206, Length = 3, Name = "kokura_shogai_chaku_kaisu_2", IsPrimaryKey = false, Comment = "小倉障害着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1209, Length = 3, Name = "kokura_shogai_chaku_kaisu_3", IsPrimaryKey = false, Comment = "小倉障害着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1212, Length = 3, Name = "kokura_shogai_chaku_kaisu_4", IsPrimaryKey = false, Comment = "小倉障害着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1215, Length = 3, Name = "kokura_shogai_chaku_kaisu_5", IsPrimaryKey = false, Comment = "小倉障害着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 1218, Length = 3, Name = "kokura_shogai_chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "小倉障害着回数着外", DataType = "CHAR" }
                                }
                            }
                        },

                        // 馬主情報
                        new NormalFieldDefinition { Position = 6343, Length = 6, Name = "banushi_code", IsPrimaryKey = false, Comment = "馬主コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 6349, Length = 64, Name = "banushi_name_hojinkaku_ari", IsPrimaryKey = false, Comment = "馬主名法人格有", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 6413, Length = 64, Name = "banushi_name_hojinkaku_nashi", IsPrimaryKey = false, Comment = "馬主名法人格無", DataType = "CHAR" },
                        
                        // 馬主本年・累計成績情報 (2回繰り返し: 本年、累計)
                        new RepeatFieldDefinition
                        {
                            Position = 6477,
                            Length = 60,
                            RepeatCount = 2,
                            Table = new TableDefinition
                            {
                                Name = "banushi_seiseki_joho",
                                Comment = "馬主本年・累計成績情報",
                                Fields = new List<FieldDefinition>
                                {
                                    new NormalFieldDefinition { Position = 1, Length = 4, Name = "settei_year", IsPrimaryKey = false, Comment = "設定年", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 5, Length = 10, Name = "hon_shokin_gokei", IsPrimaryKey = false, Comment = "本賞金合計", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 15, Length = 10, Name = "fuka_shokin_gokei", IsPrimaryKey = false, Comment = "付加賞金合計", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 25, Length = 6, Name = "chaku_kaisu_1", IsPrimaryKey = false, Comment = "着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 31, Length = 6, Name = "chaku_kaisu_2", IsPrimaryKey = false, Comment = "着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 37, Length = 6, Name = "chaku_kaisu_3", IsPrimaryKey = false, Comment = "着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 43, Length = 6, Name = "chaku_kaisu_4", IsPrimaryKey = false, Comment = "着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 49, Length = 6, Name = "chaku_kaisu_5", IsPrimaryKey = false, Comment = "着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 55, Length = 6, Name = "chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "着回数着外", DataType = "CHAR" }
                                }
                            }
                        },
                        // 生産者情報
                        new NormalFieldDefinition { Position = 6597, Length = 8, Name = "seisansha_code", IsPrimaryKey = false, Comment = "生産者コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 6605, Length = 72, Name = "seisansha_name_hojinkaku_ari", IsPrimaryKey = false, Comment = "生産者名法人格有", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 6677, Length = 72, Name = "seisansha_name_hojinkaku_nashi", IsPrimaryKey = false, Comment = "生産者名法人格無", DataType = "CHAR" },
                        
                        // 生産者本年・累計成績情報 (2回繰り返し: 本年、累計)
                        new RepeatFieldDefinition
                        {
                            Position = 6749,
                            Length = 60,
                            RepeatCount = 2,
                            Table = new TableDefinition
                            {
                                Name = "seisansha_seiseki_joho",
                                Comment = "生産者本年・累計成績情報",
                                Fields = new List<FieldDefinition>
                                {
                                    new NormalFieldDefinition { Position = 1, Length = 4, Name = "settei_year", IsPrimaryKey = false, Comment = "設定年", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 5, Length = 10, Name = "hon_shokin_gokei", IsPrimaryKey = false, Comment = "本賞金合計", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 15, Length = 10, Name = "fuka_shokin_gokei", IsPrimaryKey = false, Comment = "付加賞金合計", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 25, Length = 6, Name = "chaku_kaisu_1", IsPrimaryKey = false, Comment = "着回数1着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 31, Length = 6, Name = "chaku_kaisu_2", IsPrimaryKey = false, Comment = "着回数2着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 37, Length = 6, Name = "chaku_kaisu_3", IsPrimaryKey = false, Comment = "着回数3着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 43, Length = 6, Name = "chaku_kaisu_4", IsPrimaryKey = false, Comment = "着回数4着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 49, Length = 6, Name = "chaku_kaisu_5", IsPrimaryKey = false, Comment = "着回数5着", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 55, Length = 6, Name = "chaku_kaisu_chakugai", IsPrimaryKey = false, Comment = "着回数着外", DataType = "CHAR" }
                                }
                            }
                        }
                    }
                }
            };
            recordDefinitions.Add(ckRecord);

            // RC - レコードマスタ
            var rcRecord = new RecordDefinition
            {
                RecordTypeId = "RC",
                CreationDateField = new FieldDefinition { Position = 4, Length = 8 },
                Table = new TableDefinition
                {
                    Name = "record_master",
                    Comment = "レコードマスタ",
                    Fields = new List<FieldDefinition>
                    {
                        new NormalFieldDefinition { Position = 1, Length = 2, Name = "record_type_id", IsPrimaryKey = false, Comment = "レコード種別ID", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 3, Length = 1, Name = "data_kubun", IsPrimaryKey = false, Comment = "データ区分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 12, Length = 1, Name = "record_shikibetsu_kubun", IsPrimaryKey = true, Comment = "レコード識別区分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 13, Length = 4, Name = "kaisai_year", IsPrimaryKey = true, Comment = "開催年", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 17, Length = 4, Name = "kaisai_monthday", IsPrimaryKey = true, Comment = "開催月日", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 21, Length = 2, Name = "keibajo_code", IsPrimaryKey = true, Comment = "競馬場コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 23, Length = 2, Name = "kaisai_kai", IsPrimaryKey = true, Comment = "開催回", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 25, Length = 2, Name = "kaisai_nichime", IsPrimaryKey = true, Comment = "開催日目", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 27, Length = 2, Name = "race_number", IsPrimaryKey = true, Comment = "レース番号", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 29, Length = 4, Name = "tokubetsu_kyoso_number", IsPrimaryKey = true, Comment = "特別競走番号", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 33, Length = 60, Name = "kyoso_name_hondai", IsPrimaryKey = false, Comment = "競走名本題", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 93, Length = 1, Name = "grade_code", IsPrimaryKey = false, Comment = "グレードコード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 94, Length = 2, Name = "kyoso_shubetsu_code", IsPrimaryKey = true, Comment = "競走種別コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 96, Length = 4, Name = "kyori", IsPrimaryKey = true, Comment = "距離", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 100, Length = 2, Name = "track_code", IsPrimaryKey = true, Comment = "トラックコード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 102, Length = 1, Name = "record_kubun", IsPrimaryKey = false, Comment = "レコード区分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 103, Length = 4, Name = "record_time", IsPrimaryKey = false, Comment = "レコードタイム", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 107, Length = 1, Name = "tenko_code", IsPrimaryKey = false, Comment = "天候コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 108, Length = 1, Name = "shiba_baba_jotai_code", IsPrimaryKey = false, Comment = "芝馬場状態コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 109, Length = 1, Name = "dirt_baba_jotai_code", IsPrimaryKey = false, Comment = "ダート馬場状態コード", DataType = "CHAR" },
                        new RepeatFieldDefinition
                        {
                            Position = 110,
                            Length = 130,
                            RepeatCount = 3,
                            Table = new TableDefinition
                            {
                                Name = "record_hoji_uma_joho",
                                Comment = "レコード保持馬情報",
                                Fields = new List<FieldDefinition>
                                {
                                    new NormalFieldDefinition { Position = 1, Length = 10, Name = "ketto_toroku_number", IsPrimaryKey = false, Comment = "血統登録番号", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 11, Length = 36, Name = "uma_name", IsPrimaryKey = false, Comment = "馬名", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 47, Length = 2, Name = "uma_kigo_code", IsPrimaryKey = false, Comment = "馬記号コード", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 49, Length = 1, Name = "seibetsu_code", IsPrimaryKey = false, Comment = "性別コード", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 50, Length = 5, Name = "chokyoshi_code", IsPrimaryKey = false, Comment = "調教師コード", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 55, Length = 34, Name = "chokyoshi_name", IsPrimaryKey = false, Comment = "調教師名", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 89, Length = 3, Name = "futan_juryo", IsPrimaryKey = false, Comment = "負担重量", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 92, Length = 5, Name = "kishu_code", IsPrimaryKey = false, Comment = "騎手コード", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 97, Length = 34, Name = "kishu_name", IsPrimaryKey = false, Comment = "騎手名", DataType = "CHAR" }
                                }
                            }
                        }
                    }
                }
            };
            recordDefinitions.Add(rcRecord);

            // HC - 坂路調教
            var hcRecord = new RecordDefinition
            {
                RecordTypeId = "HC",
                CreationDateField = new FieldDefinition { Position = 4, Length = 8 },
                Table = new TableDefinition
                {
                    Name = "hanro_chokyo",
                    Comment = "坂路調教",
                    Fields = new List<FieldDefinition>
                    {
                        new NormalFieldDefinition { Position = 1, Length = 2, Name = "record_type_id", IsPrimaryKey = false, Comment = "レコード種別ID", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 3, Length = 1, Name = "data_kubun", IsPrimaryKey = false, Comment = "データ区分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 12, Length = 1, Name = "training_center_kubun", IsPrimaryKey = true, Comment = "トレセン区分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 13, Length = 8, Name = "chokyo_date", IsPrimaryKey = true, Comment = "調教年月日", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 21, Length = 4, Name = "chokyo_jikoku", IsPrimaryKey = true, Comment = "調教時刻", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 25, Length = 10, Name = "ketto_toroku_number", IsPrimaryKey = true, Comment = "血統登録番号", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 35, Length = 4, Name = "time_gokei_4_furlongs", IsPrimaryKey = false, Comment = "4ハロンタイム合計(800M～0M)", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 39, Length = 3, Name = "lap_time_800_600", IsPrimaryKey = false, Comment = "ラップタイム(800M～600M)", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 42, Length = 4, Name = "time_gokei_3_furlongs", IsPrimaryKey = false, Comment = "3ハロンタイム合計(600M～0M)", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 46, Length = 3, Name = "lap_time_600_400", IsPrimaryKey = false, Comment = "ラップタイム(600M～400M)", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 49, Length = 4, Name = "time_gokei_2_furlongs", IsPrimaryKey = false, Comment = "2ハロンタイム合計(400M～0M)", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 53, Length = 3, Name = "lap_time_400_200", IsPrimaryKey = false, Comment = "ラップタイム(400M～200M)", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 56, Length = 3, Name = "lap_time_200_0", IsPrimaryKey = false, Comment = "ラップタイム(200M～0M)", DataType = "CHAR" }
                    }
                }
            };
            recordDefinitions.Add(hcRecord);

            // HS - 競走馬市場取引価格
            var hsRecord = new RecordDefinition
            {
                RecordTypeId = "HS",
                CreationDateField = new FieldDefinition { Position = 4, Length = 8 },
                Table = new TableDefinition
                {
                    Name = "kyosoba_shijo_torihiki_kakaku",
                    Comment = "競走馬市場取引価格",
                    Fields = new List<FieldDefinition>
                    {
                        new NormalFieldDefinition { Position = 1, Length = 2, Name = "record_type_id", IsPrimaryKey = false, Comment = "レコード種別ID", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 3, Length = 1, Name = "data_kubun", IsPrimaryKey = false, Comment = "データ区分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 12, Length = 10, Name = "ketto_toroku_number", IsPrimaryKey = true, Comment = "血統登録番号", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 22, Length = 10, Name = "chichi_uma_hanshoku_toroku_number", IsPrimaryKey = false, Comment = "父馬繁殖登録番号", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 32, Length = 10, Name = "haha_uma_hanshoku_toroku_number", IsPrimaryKey = false, Comment = "母馬繁殖登録番号", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 42, Length = 4, Name = "birth_year", IsPrimaryKey = false, Comment = "生年", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 46, Length = 6, Name = "shusaisha_shijo_code", IsPrimaryKey = true, Comment = "主催者・市場コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 52, Length = 40, Name = "shusaisha_name", IsPrimaryKey = false, Comment = "主催者名称", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 92, Length = 80, Name = "shijo_name", IsPrimaryKey = false, Comment = "市場の名称", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 172, Length = 8, Name = "shijo_kaisai_start_date", IsPrimaryKey = true, Comment = "市場の開催期間(開始日)", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 180, Length = 8, Name = "shijo_kaisai_end_date", IsPrimaryKey = false, Comment = "市場の開催期間(終了日)", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 188, Length = 1, Name = "torihiki_ji_barei", IsPrimaryKey = false, Comment = "取引時の競走馬の年齢", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 189, Length = 10, Name = "torihiki_kakaku", IsPrimaryKey = false, Comment = "取引価格", DataType = "CHAR" }
                    }
                }
            };
            recordDefinitions.Add(hsRecord);

            // HY - 馬名の意味由来
            var hyRecord = new RecordDefinition
            {
                RecordTypeId = "HY",
                CreationDateField = new FieldDefinition { Position = 4, Length = 8 },
                Table = new TableDefinition
                {
                    Name = "uma_name_imi_yurai",
                    Comment = "馬名の意味由来",
                    Fields = new List<FieldDefinition>
                    {
                        new NormalFieldDefinition { Position = 1, Length = 2, Name = "record_type_id", IsPrimaryKey = false, Comment = "レコード種別ID", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 3, Length = 1, Name = "data_kubun", IsPrimaryKey = false, Comment = "データ区分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 12, Length = 10, Name = "ketto_toroku_number", IsPrimaryKey = true, Comment = "血統登録番号", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 22, Length = 36, Name = "uma_name", IsPrimaryKey = false, Comment = "馬名", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 58, Length = 64, Name = "uma_name_imi_yurai", IsPrimaryKey = false, Comment = "馬名の意味由来", DataType = "CHAR" }
                    }
                }
            };
            recordDefinitions.Add(hyRecord);

            // YS - 開催スケジュール
            var ysRecord = new RecordDefinition
            {
                RecordTypeId = "YS",
                CreationDateField = new FieldDefinition { Position = 4, Length = 8 },
                Table = new TableDefinition
                {
                    Name = "kaisai_schedule",
                    Comment = "開催スケジュール",
                    Fields = new List<FieldDefinition>
                    {
                        new NormalFieldDefinition { Position = 1, Length = 2, Name = "record_type_id", IsPrimaryKey = false, Comment = "レコード種別ID", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 3, Length = 1, Name = "data_kubun", IsPrimaryKey = false, Comment = "データ区分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 12, Length = 4, Name = "kaisai_year", IsPrimaryKey = true, Comment = "開催年", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 16, Length = 4, Name = "kaisai_monthday", IsPrimaryKey = true, Comment = "開催月日", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 20, Length = 2, Name = "keibajo_code", IsPrimaryKey = true, Comment = "競馬場コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 22, Length = 2, Name = "kaisai_kai", IsPrimaryKey = true, Comment = "開催回", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 24, Length = 2, Name = "kaisai_nichime", IsPrimaryKey = true, Comment = "開催日目", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 26, Length = 1, Name = "yobi_code", IsPrimaryKey = false, Comment = "曜日コード", DataType = "CHAR" },
                        
                        // 重賞案内 (3回繰り返し)
                        new RepeatFieldDefinition
                        {
                            Position = 27,
                            Length = 118,
                            RepeatCount = 3,
                            Table = new TableDefinition
                            {
                                Name = "jusho_annai",
                                Comment = "重賞案内",
                                Fields = new List<FieldDefinition>
                                {
                                    new NormalFieldDefinition { Position = 1, Length = 4, Name = "tokubetsu_kyoso_number", IsPrimaryKey = false, Comment = "特別競走番号", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 5, Length = 60, Name = "kyoso_name_hondai", IsPrimaryKey = false, Comment = "競走名本題", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 65, Length = 20, Name = "kyoso_name_short_10", IsPrimaryKey = false, Comment = "競走名略称10文字", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 85, Length = 12, Name = "kyoso_name_short_6", IsPrimaryKey = false, Comment = "競走名略称6文字", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 97, Length = 6, Name = "kyoso_name_short_3", IsPrimaryKey = false, Comment = "競走名略称3文字", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 103, Length = 3, Name = "jusho_kaiji", IsPrimaryKey = false, Comment = "重賞回次", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 106, Length = 1, Name = "grade_code", IsPrimaryKey = false, Comment = "グレードコード", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 107, Length = 2, Name = "kyoso_shubetsu_code", IsPrimaryKey = false, Comment = "競走種別コード", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 109, Length = 3, Name = "kyoso_kigo_code", IsPrimaryKey = false, Comment = "競走記号コード", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 112, Length = 1, Name = "juryo_shubetsu_code", IsPrimaryKey = false, Comment = "重量種別コード", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 113, Length = 4, Name = "kyori", IsPrimaryKey = false, Comment = "距離", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 117, Length = 2, Name = "track_code", IsPrimaryKey = false, Comment = "トラックコード", DataType = "CHAR" }
                                }
                            }
                        }
                    }
                }
            };
            recordDefinitions.Add(ysRecord);

            // BT - 系統情報
            var btRecord = new RecordDefinition
            {
                RecordTypeId = "BT",
                CreationDateField = new FieldDefinition { Position = 4, Length = 8 },
                Table = new TableDefinition
                {
                    Name = "keito_joho",
                    Comment = "系統情報",
                    Fields = new List<FieldDefinition>
                    {
                        new NormalFieldDefinition { Position = 1, Length = 2, Name = "record_type_id", IsPrimaryKey = false, Comment = "レコード種別ID", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 3, Length = 1, Name = "data_kubun", IsPrimaryKey = false, Comment = "データ区分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 12, Length = 10, Name = "hanshoku_toroku_number", IsPrimaryKey = true, Comment = "繁殖登録番号", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 22, Length = 30, Name = "keito_id", IsPrimaryKey = false, Comment = "系統ID", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 52, Length = 36, Name = "keito_name", IsPrimaryKey = false, Comment = "系統名", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 88, Length = 6800, Name = "keito_setumei", IsPrimaryKey = false, Comment = "系統説明", DataType = "CHAR" }
                    }
                }
            };
            recordDefinitions.Add(btRecord);

            // CS - コース情報
            var csRecord = new RecordDefinition
            {
                RecordTypeId = "CS",
                CreationDateField = new FieldDefinition { Position = 4, Length = 8 },
                Table = new TableDefinition
                {
                    Name = "course_joho",
                    Comment = "コース情報",
                    Fields = new List<FieldDefinition>
                    {
                        new NormalFieldDefinition { Position = 1, Length = 2, Name = "record_type_id", IsPrimaryKey = false, Comment = "レコード種別ID", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 3, Length = 1, Name = "data_kubun", IsPrimaryKey = false, Comment = "データ区分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 12, Length = 2, Name = "keibajo_code", IsPrimaryKey = true, Comment = "競馬場コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 14, Length = 4, Name = "kyori", IsPrimaryKey = true, Comment = "距離", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 18, Length = 2, Name = "track_code", IsPrimaryKey = true, Comment = "トラックコード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 20, Length = 8, Name = "course_kaishyu_date", IsPrimaryKey = true, Comment = "コース改修年月日", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 28, Length = 6800, Name = "course_setumei", IsPrimaryKey = false, Comment = "コース説明", DataType = "CHAR" }
                    }
                }
            };
            recordDefinitions.Add(csRecord);

            // DM - タイム型データマイニング予想
            var dmRecord = new RecordDefinition
            {
                RecordTypeId = "DM",
                CreationDateField = new FieldDefinition { Position = 4, Length = 8 },
                Table = new TableDefinition
                {
                    Name = "time_gata_data_mining_yoso",
                    Comment = "タイム型データマイニング予想",
                    Fields = new List<FieldDefinition>
                    {
                        new NormalFieldDefinition { Position = 1, Length = 2, Name = "record_type_id", IsPrimaryKey = false, Comment = "レコード種別ID", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 3, Length = 1, Name = "data_kubun", IsPrimaryKey = false, Comment = "データ区分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 12, Length = 4, Name = "kaisai_year", IsPrimaryKey = true, Comment = "開催年", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 16, Length = 4, Name = "kaisai_monthday", IsPrimaryKey = true, Comment = "開催月日", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 20, Length = 2, Name = "keibajo_code", IsPrimaryKey = true, Comment = "競馬場コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 22, Length = 2, Name = "kaisai_kai", IsPrimaryKey = true, Comment = "開催回", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 24, Length = 2, Name = "kaisai_nichime", IsPrimaryKey = true, Comment = "開催日目", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 26, Length = 2, Name = "race_number", IsPrimaryKey = true, Comment = "レース番号", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 28, Length = 4, Name = "data_sakusei_hourmin", IsPrimaryKey = false, Comment = "データ作成時分", DataType = "CHAR" },
                        
                        new RepeatFieldDefinition
                        {
                            Position = 32,
                            Length = 15,
                            RepeatCount = 18,
                            Table = new TableDefinition
                            {
                                Name = "mining_yoso",
                                Comment = "マイニング予想",
                                Fields = new List<FieldDefinition>
                                {
                                    new NormalFieldDefinition { Position = 1, Length = 2, Name = "umaban", IsPrimaryKey = false, Comment = "馬番", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 3, Length = 5, Name = "yoso_soha_time", IsPrimaryKey = false, Comment = "予想走破タイム", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 8, Length = 4, Name = "yoso_gosa_plus", IsPrimaryKey = false, Comment = "予想誤差(信頼度)＋", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 12, Length = 4, Name = "yoso_gosa_minus", IsPrimaryKey = false, Comment = "予想誤差(信頼度)－", DataType = "CHAR" }
                                }
                            }
                        }
                    }
                }
            };
            recordDefinitions.Add(dmRecord);

            // TM - 対戦型データマイニング予想
            var tmRecord = new RecordDefinition
            {
                RecordTypeId = "TM",
                CreationDateField = new FieldDefinition { Position = 4, Length = 8 },
                Table = new TableDefinition
                {
                    Name = "taisen_gata_data_mining_yoso",
                    Comment = "対戦型データマイニング予想",
                    Fields = new List<FieldDefinition>
                    {
                        new NormalFieldDefinition { Position = 1, Length = 2, Name = "record_type_id", IsPrimaryKey = false, Comment = "レコード種別ID", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 3, Length = 1, Name = "data_kubun", IsPrimaryKey = false, Comment = "データ区分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 12, Length = 4, Name = "kaisai_year", IsPrimaryKey = true, Comment = "開催年", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 16, Length = 4, Name = "kaisai_monthday", IsPrimaryKey = true, Comment = "開催月日", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 20, Length = 2, Name = "keibajo_code", IsPrimaryKey = true, Comment = "競馬場コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 22, Length = 2, Name = "kaisai_kai", IsPrimaryKey = true, Comment = "開催回", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 24, Length = 2, Name = "kaisai_nichime", IsPrimaryKey = true, Comment = "開催日目", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 26, Length = 2, Name = "race_number", IsPrimaryKey = true, Comment = "レース番号", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 28, Length = 4, Name = "data_sakusei_hourmin", IsPrimaryKey = false, Comment = "データ作成時分", DataType = "CHAR" },
                        new RepeatFieldDefinition
                        {
                            Position = 32,
                            Length = 6,
                            RepeatCount = 18,
                            Table = new TableDefinition
                            {
                                Name = "mining_yoso_uma",
                                Comment = "マイニング予想馬毎情報",
                                Fields = new List<FieldDefinition>
                                {
                                    new NormalFieldDefinition { Position = 1, Length = 2, Name = "umaban", IsPrimaryKey = false, Comment = "馬番", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 3, Length = 4, Name = "yosoku_score", IsPrimaryKey = false, Comment = "予測スコア", DataType = "CHAR" }
                                }
                            }
                        }
                    }
                }
            };
            recordDefinitions.Add(tmRecord);

            // WF - 重勝式(WIN5)
            var wfRecord = new RecordDefinition
            {
                RecordTypeId = "WF",
                CreationDateField = new FieldDefinition { Position = 4, Length = 8 },
                Table = new TableDefinition
                {
                    Name = "win5_joho",
                    Comment = "重勝式(WIN5)情報",
                    Fields = new List<FieldDefinition>
                    {
                        new NormalFieldDefinition { Position = 1, Length = 2, Name = "record_type_id", IsPrimaryKey = false, Comment = "レコード種別ID", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 3, Length = 1, Name = "data_kubun", IsPrimaryKey = false, Comment = "データ区分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 12, Length = 4, Name = "kaisai_year", IsPrimaryKey = true, Comment = "開催年", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 16, Length = 4, Name = "kaisai_monthday", IsPrimaryKey = true, Comment = "開催月日", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 20, Length = 2, Name = "yobi_1", IsPrimaryKey = false, Comment = "予備", DataType = "CHAR" },
                        
                        // 重勝式対象レース情報 (5回繰り返し)
                        new RepeatFieldDefinition
                        {
                            Position = 22,
                            Length = 8,
                            RepeatCount = 5,
                            Table = new TableDefinition
                            {
                                Name = "win5_taisho_race_joho",
                                Comment = "重勝式対象レース情報",
                                Fields = new List<FieldDefinition>
                                {
                                    new NormalFieldDefinition { Position = 1, Length = 2, Name = "keibajo_code", IsPrimaryKey = false, Comment = "競馬場コード", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 3, Length = 2, Name = "kaisai_kai", IsPrimaryKey = false, Comment = "開催回", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 5, Length = 2, Name = "kaisai_nichime", IsPrimaryKey = false, Comment = "開催日目", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 7, Length = 2, Name = "race_number", IsPrimaryKey = false, Comment = "レース番号", DataType = "CHAR" }
                                }
                            }
                        },
                        
                        new NormalFieldDefinition { Position = 62, Length = 6, Name = "yobi_2", IsPrimaryKey = false, Comment = "予備", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 68, Length = 11, Name = "win5_hatsubai_hyosu", IsPrimaryKey = false, Comment = "重勝式発売票数", DataType = "CHAR" },
                        
                        // 有効票数情報 (5回繰り返し)
                        new NormalFieldDefinition { Position = 79, Length = 11, Name = "yuko_hyosu_1", IsPrimaryKey = false, Comment = "有効票数1", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 90, Length = 11, Name = "yuko_hyosu_2", IsPrimaryKey = false, Comment = "有効票数2", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 101, Length = 11, Name = "yuko_hyosu_3", IsPrimaryKey = false, Comment = "有効票数3", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 112, Length = 11, Name = "yuko_hyosu_4", IsPrimaryKey = false, Comment = "有効票数4", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 123, Length = 11, Name = "yuko_hyosu_5", IsPrimaryKey = false, Comment = "有効票数5", DataType = "CHAR" },
                        
                        new NormalFieldDefinition { Position = 134, Length = 1, Name = "henkan_flag", IsPrimaryKey = false, Comment = "返還フラグ", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 135, Length = 1, Name = "fuseirtsu_flag", IsPrimaryKey = false, Comment = "不成立フラグ", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 136, Length = 1, Name = "tekichu_nashi_flag", IsPrimaryKey = false, Comment = "的中無フラグ", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 137, Length = 15, Name = "carryover_kingaku_shoki", IsPrimaryKey = false, Comment = "キャリーオーバー金額初期", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 152, Length = 15, Name = "carryover_kingaku_zandaka", IsPrimaryKey = false, Comment = "キャリーオーバー金額残高", DataType = "CHAR" },
                        
                        // 重勝式払戻情報 (243回繰り返し)
                        new RepeatFieldDefinition
                        {
                            Position = 167,
                            Length = 29,
                            RepeatCount = 243,
                            Table = new TableDefinition
                            {
                                Name = "win5_haraimodoshi_joho",
                                Comment = "重勝式払戻情報",
                                Fields = new List<FieldDefinition>
                                {
                                    new NormalFieldDefinition { Position = 1, Length = 10, Name = "kumiban", IsPrimaryKey = false, Comment = "組番", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 11, Length = 9, Name = "win5_haraimodoshi_kin", IsPrimaryKey = false, Comment = "重勝式払戻金", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 20, Length = 10, Name = "tekichu_hyosu", IsPrimaryKey = false, Comment = "的中票数", DataType = "CHAR" }
                                }
                            }
                        }
                    }
                }
            };
            recordDefinitions.Add(wfRecord);

            // JG - 競走馬除外情報
            var jgRecord = new RecordDefinition
            {
                RecordTypeId = "JG",
                CreationDateField = new FieldDefinition { Position = 4, Length = 8 },
                Table = new TableDefinition
                {
                    Name = "kyosoba_jogai_joho",
                    Comment = "競走馬除外情報",
                    Fields = new List<FieldDefinition>
                    {
                        new NormalFieldDefinition { Position = 1, Length = 2, Name = "record_type_id", IsPrimaryKey = false, Comment = "レコード種別ID", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 3, Length = 1, Name = "data_kubun", IsPrimaryKey = false, Comment = "データ区分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 12, Length = 4, Name = "kaisai_year", IsPrimaryKey = true, Comment = "開催年", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 16, Length = 4, Name = "kaisai_monthday", IsPrimaryKey = true, Comment = "開催月日", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 20, Length = 2, Name = "keibajo_code", IsPrimaryKey = true, Comment = "競馬場コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 22, Length = 2, Name = "kaisai_kai", IsPrimaryKey = true, Comment = "開催回", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 24, Length = 2, Name = "kaisai_nichime", IsPrimaryKey = true, Comment = "開催日目", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 26, Length = 2, Name = "race_number", IsPrimaryKey = true, Comment = "レース番号", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 28, Length = 10, Name = "ketto_toroku_number", IsPrimaryKey = true, Comment = "血統登録番号", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 38, Length = 36, Name = "uma_name", IsPrimaryKey = false, Comment = "馬名", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 74, Length = 3, Name = "shutsuba_tohyo_uketsuke_junban", IsPrimaryKey = true, Comment = "出馬投票受付順番", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 77, Length = 1, Name = "shusso_kubun", IsPrimaryKey = false, Comment = "出走区分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 78, Length = 1, Name = "jogai_jotai_kubun", IsPrimaryKey = false, Comment = "除外状態区分", DataType = "CHAR" }
                    }
                }
            };
            recordDefinitions.Add(jgRecord);

            // WC - ウッドチップ調教
            var wcRecord = new RecordDefinition
            {
                RecordTypeId = "WC",
                CreationDateField = new FieldDefinition { Position = 4, Length = 8 },
                Table = new TableDefinition
                {
                    Name = "wood_chip_chokyo",
                    Comment = "ウッドチップ調教",
                    Fields = new List<FieldDefinition>
                    {
                        new NormalFieldDefinition { Position = 1, Length = 2, Name = "record_type_id", IsPrimaryKey = false, Comment = "レコード種別ID", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 3, Length = 1, Name = "data_kubun", IsPrimaryKey = false, Comment = "データ区分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 12, Length = 1, Name = "training_center_kubun", IsPrimaryKey = true, Comment = "トレセン区分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 13, Length = 8, Name = "chokyo_date", IsPrimaryKey = true, Comment = "調教年月日", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 21, Length = 4, Name = "chokyo_jikoku", IsPrimaryKey = true, Comment = "調教時刻", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 25, Length = 10, Name = "ketto_toroku_number", IsPrimaryKey = true, Comment = "血統登録番号", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 35, Length = 1, Name = "course", IsPrimaryKey = false, Comment = "コース", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 36, Length = 1, Name = "baba_mawari", IsPrimaryKey = false, Comment = "馬場周り", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 37, Length = 1, Name = "yobi", IsPrimaryKey = false, Comment = "予備", DataType = "CHAR" },
                        
                        // 10ハロン2000M
                        new NormalFieldDefinition { Position = 38, Length = 4, Name = "time_gokei_10_furlongs", IsPrimaryKey = false, Comment = "10ハロンタイム合計(2000M～0M)", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 42, Length = 3, Name = "lap_time_2000_1800", IsPrimaryKey = false, Comment = "ラップタイム(2000M～1800M)", DataType = "CHAR" },
                        
                        // 9ハロン1800M
                        new NormalFieldDefinition { Position = 45, Length = 4, Name = "time_gokei_9_furlongs", IsPrimaryKey = false, Comment = "9ハロンタイム合計(1800M～0M)", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 49, Length = 3, Name = "lap_time_1800_1600", IsPrimaryKey = false, Comment = "ラップタイム(1800M～1600M)", DataType = "CHAR" },
                        
                        // 8ハロン1600M
                        new NormalFieldDefinition { Position = 52, Length = 4, Name = "time_gokei_8_furlongs", IsPrimaryKey = false, Comment = "8ハロンタイム合計(1600M～0M)", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 56, Length = 3, Name = "lap_time_1600_1400", IsPrimaryKey = false, Comment = "ラップタイム(1600M～1400M)", DataType = "CHAR" },
                        
                        // 7ハロン1400M
                        new NormalFieldDefinition { Position = 59, Length = 4, Name = "time_gokei_7_furlongs", IsPrimaryKey = false, Comment = "7ハロンタイム合計(1400M～0M)", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 63, Length = 3, Name = "lap_time_1400_1200", IsPrimaryKey = false, Comment = "ラップタイム(1400M～1200M)", DataType = "CHAR" },
                        
                        // 6ハロン1200M
                        new NormalFieldDefinition { Position = 66, Length = 4, Name = "time_gokei_6_furlongs", IsPrimaryKey = false, Comment = "6ハロンタイム合計(1200M～0M)", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 70, Length = 3, Name = "lap_time_1200_1000", IsPrimaryKey = false, Comment = "ラップタイム(1200M～1000M)", DataType = "CHAR" },
                        
                        // 5ハロン1000M
                        new NormalFieldDefinition { Position = 73, Length = 4, Name = "time_gokei_5_furlongs", IsPrimaryKey = false, Comment = "5ハロンタイム合計(1000M～0M)", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 77, Length = 3, Name = "lap_time_1000_800", IsPrimaryKey = false, Comment = "ラップタイム(1000M～800M)", DataType = "CHAR" },
                        
                        // 4ハロン800M
                        new NormalFieldDefinition { Position = 80, Length = 4, Name = "time_gokei_4_furlongs", IsPrimaryKey = false, Comment = "4ハロンタイム合計(800M～0M)", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 84, Length = 3, Name = "lap_time_800_600", IsPrimaryKey = false, Comment = "ラップタイム(800M～600M)", DataType = "CHAR" },
                        
                        // 3ハロン600M
                        new NormalFieldDefinition { Position = 87, Length = 4, Name = "time_gokei_3_furlongs", IsPrimaryKey = false, Comment = "3ハロンタイム合計(600M～0M)", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 91, Length = 3, Name = "lap_time_600_400", IsPrimaryKey = false, Comment = "ラップタイム(600M～400M)", DataType = "CHAR" },
                        
                        // 2ハロン400M
                        new NormalFieldDefinition { Position = 94, Length = 4, Name = "time_gokei_2_furlongs", IsPrimaryKey = false, Comment = "2ハロンタイム合計(400M～0M)", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 98, Length = 3, Name = "lap_time_400_200", IsPrimaryKey = false, Comment = "ラップタイム(400M～200M)", DataType = "CHAR" },
                        
                        // 1ハロン200M
                        new NormalFieldDefinition { Position = 101, Length = 3, Name = "lap_time_200_0", IsPrimaryKey = false, Comment = "ラップタイム(200M～0M)", DataType = "CHAR" }
                    }
                }
            };
            recordDefinitions.Add(wcRecord);

            // WH - 馬体重
            var whRecord = new RecordDefinition
            {
                RecordTypeId = "WH",
                CreationDateField = new FieldDefinition { Position = 4, Length = 8 },
                Table = new TableDefinition
                {
                    Name = "bataiju",
                    Comment = "馬体重",
                    Fields = new List<FieldDefinition>
                    {
                        new NormalFieldDefinition { Position = 1, Length = 2, Name = "record_type_id", IsPrimaryKey = false, Comment = "レコード種別ID", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 3, Length = 1, Name = "data_kubun", IsPrimaryKey = false, Comment = "データ区分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 12, Length = 4, Name = "kaisai_year", IsPrimaryKey = true, Comment = "開催年", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 16, Length = 4, Name = "kaisai_monthday", IsPrimaryKey = true, Comment = "開催月日", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 20, Length = 2, Name = "keibajo_code", IsPrimaryKey = true, Comment = "競馬場コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 22, Length = 2, Name = "kaisai_kai", IsPrimaryKey = true, Comment = "開催回", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 24, Length = 2, Name = "kaisai_nichime", IsPrimaryKey = true, Comment = "開催日目", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 26, Length = 2, Name = "race_number", IsPrimaryKey = true, Comment = "レース番号", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 28, Length = 8, Name = "happyo_monthday_hourmin", IsPrimaryKey = false, Comment = "発表月日時分", DataType = "CHAR" },
                        new RepeatFieldDefinition
                        {
                            Position = 36,
                            Length = 45,
                            RepeatCount = 18,
                            Table = new TableDefinition
                            {
                                Name = "bataiju_joho",
                                Comment = "馬体重情報",
                                Fields = new List<FieldDefinition>
                                {
                                    new NormalFieldDefinition { Position = 1, Length = 2, Name = "umaban", IsPrimaryKey = false, Comment = "馬番", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 3, Length = 36, Name = "uma_name", IsPrimaryKey = false, Comment = "馬名", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 39, Length = 3, Name = "bataiju", IsPrimaryKey = false, Comment = "馬体重", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 42, Length = 1, Name = "zogen_fugo", IsPrimaryKey = false, Comment = "増減符号", DataType = "CHAR" },
                                    new NormalFieldDefinition { Position = 43, Length = 3, Name = "zogen_sa", IsPrimaryKey = false, Comment = "増減差", DataType = "CHAR" }
                                }
                            }
                        }
                    }
                }
            };
            recordDefinitions.Add(whRecord);

            // WE - 天候馬場状態
            var weRecord = new RecordDefinition
            {
                RecordTypeId = "WE",
                CreationDateField = new FieldDefinition { Position = 4, Length = 8 },
                Table = new TableDefinition
                {
                    Name = "tenko_baba_jotai",
                    Comment = "天候馬場状態",
                    Fields = new List<FieldDefinition>
                    {
                        new NormalFieldDefinition { Position = 1, Length = 2, Name = "record_type_id", IsPrimaryKey = false, Comment = "レコード種別ID", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 3, Length = 1, Name = "data_kubun", IsPrimaryKey = false, Comment = "データ区分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 12, Length = 4, Name = "kaisai_year", IsPrimaryKey = true, Comment = "開催年", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 16, Length = 4, Name = "kaisai_monthday", IsPrimaryKey = true, Comment = "開催月日", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 20, Length = 2, Name = "keibajo_code", IsPrimaryKey = true, Comment = "競馬場コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 22, Length = 2, Name = "kaisai_kai", IsPrimaryKey = true, Comment = "開催回", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 24, Length = 2, Name = "kaisai_nichime", IsPrimaryKey = true, Comment = "開催日目", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 26, Length = 8, Name = "happyo_monthday_time", IsPrimaryKey = true, Comment = "発表月日時分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 34, Length = 1, Name = "henko_shikibetsu", IsPrimaryKey = false, Comment = "変更識別", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 35, Length = 1, Name = "tenko_code", IsPrimaryKey = false, Comment = "天候状態", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 36, Length = 1, Name = "shiba_baba_jotai_code", IsPrimaryKey = false, Comment = "馬場状態・芝", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 37, Length = 1, Name = "dirt_baba_jotai_code", IsPrimaryKey = false, Comment = "馬場状態・ダート", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 38, Length = 1, Name = "prev_tenko_code", IsPrimaryKey = false, Comment = "変更前天候状態", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 39, Length = 1, Name = "prev_shiba_baba_jotai_code", IsPrimaryKey = false, Comment = "変更前馬場状態・芝", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 40, Length = 1, Name = "prev_dirt_baba_jotai_code", IsPrimaryKey = false, Comment = "変更前馬場状態・ダート", DataType = "CHAR" }
                    }
                }
            };
            recordDefinitions.Add(weRecord);

            // AV - 出走取消・競走除外
            var avRecord = new RecordDefinition
            {
                RecordTypeId = "AV",
                CreationDateField = new FieldDefinition { Position = 4, Length = 8 },
                Table = new TableDefinition
                {
                    Name = "shusso_torikeshi_kyoso_jogai",
                    Comment = "出走取消・競走除外",
                    Fields = new List<FieldDefinition>
                    {
                        new NormalFieldDefinition { Position = 1, Length = 2, Name = "record_type_id", IsPrimaryKey = false, Comment = "レコード種別ID", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 3, Length = 1, Name = "data_kubun", IsPrimaryKey = false, Comment = "データ区分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 12, Length = 4, Name = "kaisai_year", IsPrimaryKey = true, Comment = "開催年", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 16, Length = 4, Name = "kaisai_monthday", IsPrimaryKey = true, Comment = "開催月日", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 20, Length = 2, Name = "keibajo_code", IsPrimaryKey = true, Comment = "競馬場コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 22, Length = 2, Name = "kaisai_kai", IsPrimaryKey = true, Comment = "開催回", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 24, Length = 2, Name = "kaisai_nichime", IsPrimaryKey = true, Comment = "開催日目", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 26, Length = 2, Name = "race_number", IsPrimaryKey = true, Comment = "レース番号", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 28, Length = 8, Name = "happyo_monthday_hourmin", IsPrimaryKey = false, Comment = "発表月日時分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 36, Length = 2, Name = "umaban", IsPrimaryKey = true, Comment = "馬番", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 38, Length = 36, Name = "uma_name", IsPrimaryKey = false, Comment = "馬名", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 74, Length = 3, Name = "jiyu_kubun", IsPrimaryKey = false, Comment = "事由区分", DataType = "CHAR" }
                    }
                }
            };
            recordDefinitions.Add(avRecord);

            // JC - 騎手変更
            var jcRecord = new RecordDefinition
            {
                RecordTypeId = "JC",
                CreationDateField = new FieldDefinition { Position = 4, Length = 8 },
                Table = new TableDefinition
                {
                    Name = "kishu_henko",
                    Comment = "騎手変更",
                    Fields = new List<FieldDefinition>
                    {
                        new NormalFieldDefinition { Position = 1, Length = 2, Name = "record_type_id", IsPrimaryKey = false, Comment = "レコード種別ID", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 3, Length = 1, Name = "data_kubun", IsPrimaryKey = false, Comment = "データ区分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 12, Length = 4, Name = "kaisai_year", IsPrimaryKey = true, Comment = "開催年", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 16, Length = 4, Name = "kaisai_monthday", IsPrimaryKey = true, Comment = "開催月日", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 20, Length = 2, Name = "keibajo_code", IsPrimaryKey = true, Comment = "競馬場コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 22, Length = 2, Name = "kaisai_kai", IsPrimaryKey = true, Comment = "開催回", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 24, Length = 2, Name = "kaisai_nichime", IsPrimaryKey = true, Comment = "開催日目", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 26, Length = 2, Name = "race_number", IsPrimaryKey = true, Comment = "レース番号", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 28, Length = 8, Name = "happyo_monthday_hourmin", IsPrimaryKey = false, Comment = "発表月日時分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 36, Length = 2, Name = "umaban", IsPrimaryKey = true, Comment = "馬番", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 38, Length = 36, Name = "uma_name", IsPrimaryKey = false, Comment = "馬名", DataType = "CHAR" },
                        // 変更後情報
                        new NormalFieldDefinition { Position = 74, Length = 3, Name = "futan_juryo", IsPrimaryKey = false, Comment = "負担重量(変更後)", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 77, Length = 5, Name = "kishu_code", IsPrimaryKey = false, Comment = "騎手コード(変更後)", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 82, Length = 34, Name = "kishu_name", IsPrimaryKey = false, Comment = "騎手名(変更後)", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 116, Length = 1, Name = "kishu_minarai_code", IsPrimaryKey = false, Comment = "騎手見習コード(変更後)", DataType = "CHAR" },
                        // 変更前情報
                        new NormalFieldDefinition { Position = 117, Length = 3, Name = "prev_futan_juryo", IsPrimaryKey = false, Comment = "負担重量(変更前)", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 120, Length = 5, Name = "prev_kishu_code", IsPrimaryKey = false, Comment = "騎手コード(変更前)", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 125, Length = 34, Name = "prev_kishu_name", IsPrimaryKey = false, Comment = "騎手名(変更前)", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 159, Length = 1, Name = "prev_kishu_minarai_code", IsPrimaryKey = false, Comment = "騎手見習コード(変更前)", DataType = "CHAR" }
                    }
                }
            };
            recordDefinitions.Add(jcRecord);

            // TC - 発走時刻変更
            var tcRecord = new RecordDefinition
            {
                RecordTypeId = "TC",
                CreationDateField = new FieldDefinition { Position = 4, Length = 8 },
                Table = new TableDefinition
                {
                    Name = "hasso_jikoku_henko",
                    Comment = "発走時刻変更",
                    Fields = new List<FieldDefinition>
                    {
                        new NormalFieldDefinition { Position = 1, Length = 2, Name = "record_type_id", IsPrimaryKey = false, Comment = "レコード種別ID", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 3, Length = 1, Name = "data_kubun", IsPrimaryKey = false, Comment = "データ区分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 12, Length = 4, Name = "kaisai_year", IsPrimaryKey = true, Comment = "開催年", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 16, Length = 4, Name = "kaisai_monthday", IsPrimaryKey = true, Comment = "開催月日", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 20, Length = 2, Name = "keibajo_code", IsPrimaryKey = true, Comment = "競馬場コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 22, Length = 2, Name = "kaisai_kai", IsPrimaryKey = true, Comment = "開催回", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 24, Length = 2, Name = "kaisai_nichime", IsPrimaryKey = true, Comment = "開催日目", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 26, Length = 2, Name = "race_number", IsPrimaryKey = true, Comment = "レース番号", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 28, Length = 8, Name = "happyo_monthday_hourmin", IsPrimaryKey = false, Comment = "発表月日時分", DataType = "CHAR" },
                        // 変更後情報
                        new NormalFieldDefinition { Position = 36, Length = 4, Name = "hasso_jikoku", IsPrimaryKey = false, Comment = "発走時刻(変更後)", DataType = "CHAR" },
                        // 変更前情報
                        new NormalFieldDefinition { Position = 40, Length = 4, Name = "prev_hasso_jikoku", IsPrimaryKey = false, Comment = "発走時刻(変更前)", DataType = "CHAR" }
                    }
                }
            };
            recordDefinitions.Add(tcRecord);

            // CC - コース変更
            var ccRecord = new RecordDefinition
            {
                RecordTypeId = "CC",
                CreationDateField = new FieldDefinition { Position = 4, Length = 8 },
                Table = new TableDefinition
                {
                    Name = "course_henkou",
                    Comment = "コース変更",
                    Fields = new List<FieldDefinition>
                    {
                        new NormalFieldDefinition { Position = 1, Length = 2, Name = "record_type_id", IsPrimaryKey = false, Comment = "レコード種別ID", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 3, Length = 1, Name = "data_kubun", IsPrimaryKey = false, Comment = "データ区分", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 12, Length = 4, Name = "kaisai_year", IsPrimaryKey = true, Comment = "開催年", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 16, Length = 4, Name = "kaisai_monthday", IsPrimaryKey = true, Comment = "開催月日", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 20, Length = 2, Name = "keibajo_code", IsPrimaryKey = true, Comment = "競馬場コード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 22, Length = 2, Name = "kaisai_kai", IsPrimaryKey = true, Comment = "開催回", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 24, Length = 2, Name = "kaisai_nichime", IsPrimaryKey = true, Comment = "開催日目", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 26, Length = 2, Name = "race_number", IsPrimaryKey = true, Comment = "レース番号", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 28, Length = 8, Name = "happyo_monthday_hourmin", IsPrimaryKey = false, Comment = "発表月日時分", DataType = "CHAR" },
                        // 変更後情報
                        new NormalFieldDefinition { Position = 36, Length = 4, Name = "kyori", IsPrimaryKey = false, Comment = "変更後距離", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 40, Length = 2, Name = "track_code", IsPrimaryKey = false, Comment = "変更後トラックコード", DataType = "CHAR" },
                        // 変更前情報
                        new NormalFieldDefinition { Position = 42, Length = 4, Name = "prev_kyori", IsPrimaryKey = false, Comment = "変更前距離", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 46, Length = 2, Name = "prev_track_code", IsPrimaryKey = false, Comment = "変更前トラックコード", DataType = "CHAR" },
                        new NormalFieldDefinition { Position = 48, Length = 1, Name = "jiyu_kubun", IsPrimaryKey = false, Comment = "事由区分", DataType = "CHAR" }
                    }
                }
            };
            recordDefinitions.Add(ccRecord);

            Console.WriteLine($"Record definitions created successfully. Total record types: {recordDefinitions.Count}");
        }
    }
}

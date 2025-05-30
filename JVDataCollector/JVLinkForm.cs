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
        private string[] commandLineArgs;
        private string connectionString = "Host=localhost;Database=horgues3;Username=postgres;Password=postgres";

        public JVLinkForm(string[] args)
        {
            InitializeComponent();
            commandLineArgs = args;
        }

        private void JVLinkForm_Load(object sender, EventArgs e)
        {
            try
            {
                Console.WriteLine();
                Console.WriteLine($"JVDataCollector started at {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
                Console.WriteLine();

                if (commandLineArgs.Length > 0)
                {
                    string command = commandLineArgs[0].ToLower();
                    Console.WriteLine($"Executing command: {command}");
                    Console.WriteLine();

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

                Console.WriteLine();
                Console.WriteLine($"JVDataCollector completed successfully at {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
                Console.WriteLine();
            }
            catch (Exception ex)
            {
                Console.WriteLine();
                Console.WriteLine($"Error: {ex.Message}");
                Console.WriteLine($"Stack Trace:");
                Console.WriteLine(ex.StackTrace);
                Console.WriteLine($"JVDataCollector failed at {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
                Console.WriteLine();
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

            // TODO: JVStatus, JVRead を行う 

            // TODO: lastFileTimestamp をデータベースに保存する

            axJVLink1.JVClose();
            Console.WriteLine($"JV data processing completed.");
            Console.WriteLine();
        }

        private void CreateDatabaseTables()
        {
            Console.WriteLine("Creating database tables...");
            
            using (var connection = new NpgsqlConnection(connectionString))
            {
                connection.Open();
                
                string[] tableCreationSqls = BuildTableCreationSqls();
                
                foreach (string sql in tableCreationSqls)
                {
                    using (var command = new NpgsqlCommand(sql, connection))
                    {
                        command.ExecuteNonQuery();
                    }
                }
            }
            
            Console.WriteLine("Database tables created successfully.");
            Console.WriteLine();
        }

        private string[] BuildTableCreationSqls()
        {
            var sqls = new List<string>();
            StringBuilder sb = new StringBuilder();

            // TODO: CREATE TABLE を作成

            // XXXテーブル
            sb.Clear();
            sb.AppendLine(";");
            sqls.Add(sb.ToString());

            return sqls.ToArray();
        }

        private void ExecuteSetup()
        {
            CreateDatabaseTables();

            Console.WriteLine("Initializing JVLink...");
            int result = axJVLink1.JVInit("SA000000/SD000004");
            if (result != 0)
            {
                throw new InvalidOperationException($"JVInit failed with error code: {result}");
            }
            Console.WriteLine("JVLink initialized successfully.");
            Console.WriteLine();

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

        private void ExecuteUpdate()
        {
            throw new NotImplementedException("Update functionality is not yet implemented");
        }
    }
}
